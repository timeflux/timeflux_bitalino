import struct
import time
import numpy as np
from bitalino import BITalino, ExceptionCode
from timeflux.core.exceptions import WorkerInterrupt
from timeflux.core.node import Node


class Bitalino(Node):

    """BITalino driver.

    This node connects to a BITalino device and streams data at a provided rate.
    It is based on the original BITalino Python library, with some performance
    improvements and careful timestamping.

    Two output streams are provided. The default output is the data read from the
    analog and digital channels. The ``o_offsets`` output provides continuous offsets
    between the local time and the estimated device time. This enables drift correction
    to be performed during post-processing, although no significant drift has been
    observed during testing.

    Attributes:
        o (Port): BITalino data, provides DataFrame.
        o_offsets (Port): Time offsets, provide DataFrame.

    Args:
        port (string): The serial port.
            e.g. ``COM3`` on Windows;  ``/dev/tty.bitalino-DevB`` on MacOS;
            ``/dev/ttyUSB0`` on GNU/Linux.
        rate (int): The device rate in Hz.
            Possible values: ``1``, ``10``, ``100``, ``1000``. Default: ``1000``.
        channels (tupple): The analog channels to read from.
            Default: ``('A1', 'A2', 'A3', 'A4', 'A5', 'A6')``.

    Example:
        .. literalinclude:: /../examples/bitalino.yaml
           :language: yaml

    Notes:

    .. attention::

        Make sure to set your graph rate to an high-enough value, otherwise the device
        internal buffer may saturate, and data may be lost. A 30Hz graph rate is
        recommended for a 1000Hz device rate.

    """

    def __init__(self, port, rate=1000, channels=("A1", "A2", "A3", "A4", "A5", "A6")):

        # Check port
        if not port.startswith("/dev/") and not port.startswith("COM"):
            raise ValueError(f"Invalid serial port: {port}")

        # Check rate
        if rate not in (1, 10, 100, 1000):
            raise ValueError(f"Invalid rate: {rate}")

        # Check channels
        unique_channels = set(channels)
        analog_channels = ["A1", "A2", "A3", "A4", "A5", "A6"]
        channels = []
        for channel_num, channel_name in enumerate(analog_channels):
            if channel_name in unique_channels:
                channels.append(channel_num)

        # Set column names
        # Sequence number and numeric channels are always present
        self.columns = ["SEQ", "I1", "I2", "O1", "O2"]
        # Add required analog channels
        for channel in channels:
            self.columns.append(analog_channels[channel])

        # Compute the sample size in bytes
        self.channel_count = len(channels)
        if self.channel_count <= 4:
            self.sample_size = int(np.ceil((12.0 + 10.0 * self.channel_count) / 8.0))
        else:
            self.sample_size = int(
                np.ceil((52.0 + 6.0 * (self.channel_count - 4)) / 8.0)
            )

        # Connect to BITalino
        try:
            self.device = BITalino(port)
        except UnicodeDecodeError:
            # This can happen after an internal buffer overflow.
            # The solution seems to power off the device and repair.
            raise WorkerInterrupt("Unstable state. Could not connect.")
        except Exception as e:
            raise WorkerInterrupt(e)

        # Set battery threshold
        # The red led will light up at 5-10%
        self.device.battery(30)

        # Read BITalino version
        self.logger.info(self.device.version())

        # Read state and show battery level
        # http://forum.bitalino.com/viewtopic.php?t=448
        state = self.device.state()
        battery = round(1 + (state["battery"] - 511) * ((99 - 1) / (645 - 511)), 2)
        self.logger.info("Battery: %.2f%%", battery)

        # Start Acquisition
        self.device.start(rate, channels)

        # Initialize counters for timestamp indices and continuity checks
        self.last_sample_counter = 15
        self.time_device = np.datetime64(int(time.time() * 1e6), "us")
        self.time_local = self.time_device
        self.time_delta = np.timedelta64(int(1000 / rate), "ms")

        # Set meta
        self.meta = {"rate": rate}

    def update(self):
        # Send BITalino data
        data, timestamps = self._read_all()
        self.o.set(data, timestamps, self.columns, self.meta)
        # Send time offsets
        if len(timestamps) > 0:
            offset = (self.time_local - self.time_device).astype(int)
            self.o_offsets.set(
                [[self.time_device, offset]],
                [self.time_local],
                ["time_device", "time_offset"],
            )

    def _read_all(self):

        """Read all available data"""

        # Make sure the device is in aquisition mode
        if not self.device.started:
            raise Exception(ExceptionCode.DEVICE_NOT_IN_ACQUISITION)

        # We only support serial connections
        if not self.device.serial:
            raise Exception("Device must be opened in serial mode.")

        # Check buffer size and limits
        buffer_size = self.device.socket.in_waiting
        if buffer_size == 1020:
            # The device buffer can hold up to 1020 bytes
            self.logger.warn(
                "OS serial buffer saturated. Increase graph rate or decrease device rate."
            )

        # Compute the maximum number of samples we can get
        sample_count = int(buffer_size / self.sample_size)

        # Infer timestamps from sample count and rate
        # Will fail dramatically if too much packets are lost
        # Tests show that there is no significant drift during a 2-hour session
        start = self.time_device
        stop = start + self.time_delta * sample_count
        self.time_device = stop
        timestamps = np.arange(start, stop, self.time_delta)
        self.time_local = np.datetime64(int(time.time() * 1e6), "us")

        # Infer timestamps from local time and rate
        # /!\ Not monotonic
        # stop = np.datetime64(int(time.time() * 1e6), 'us')
        # start = stop - (sample_count * self.time_delta)
        # timestamps = np.arange(start, stop, self.time_delta)

        # Read raw samples from device
        raw = self.device.socket.read(sample_count * self.sample_size)

        # Initialize the output matrix
        data = np.full((sample_count, 5 + self.channel_count), np.nan)

        # Parse the raw data
        # http://bitalino.com/datasheets/REVOLUTION_MCU_Block_Datasheet.pdf
        for sample_number in range(sample_count):

            # Extract sample
            start = sample_number * self.sample_size
            stop = start + self.sample_size
            sample = list(struct.unpack(self.sample_size * "B ", raw[start:stop]))

            # Is the sample corrupted?
            crc = sample[-1] & 0x0F
            sample[-1] = sample[-1] & 0xF0
            x = 0
            for i in range(self.sample_size):
                for bit in range(7, -1, -1):
                    x = x << 1
                    if x & 0x10:
                        x = x ^ 0x03
                    x = x ^ ((sample[i] >> bit) & 0x01)
            if crc != x & 0x0F:
                self.logger.warn("Checksum failed.")
                continue

            # Parse sample
            data[sample_number, 0] = sample[-1] >> 4
            data[sample_number, 1] = sample[-2] >> 7 & 0x01
            data[sample_number, 2] = sample[-2] >> 6 & 0x01
            data[sample_number, 3] = sample[-2] >> 5 & 0x01
            data[sample_number, 4] = sample[-2] >> 4 & 0x01
            if self.channel_count > 0:
                data[sample_number, 5] = ((sample[-2] & 0x0F) << 6) | (sample[-3] >> 2)
            if self.channel_count > 1:
                data[sample_number, 6] = ((sample[-3] & 0x03) << 8) | sample[-4]
            if self.channel_count > 2:
                data[sample_number, 7] = (sample[-5] << 2) | (sample[-6] >> 6)
            if self.channel_count > 3:
                data[sample_number, 8] = ((sample[-6] & 0x3F) << 4) | (sample[-7] >> 4)
            if self.channel_count > 4:
                data[sample_number, 9] = ((sample[-7] & 0x0F) << 2) | (sample[-8] >> 6)
            if self.channel_count > 5:
                data[sample_number, 10] = sample[-8] & 0x3F

            # Did we miss any sample?
            # Check for discontinuity in the internal sample counter, encoded to 4 bits.
            sample_counter = data[sample_number, 0]
            if sample_counter == self.last_sample_counter + 1:
                pass
            elif sample_counter == 0 and self.last_sample_counter == 15:
                pass
            else:
                self.logger.warn("Missed sample.")
            self.last_sample_counter = sample_counter

        return data, timestamps

    def terminate(self):
        self.device.stop()
        self.device.close()
