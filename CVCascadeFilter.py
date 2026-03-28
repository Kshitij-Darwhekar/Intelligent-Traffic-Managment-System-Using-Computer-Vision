class CVCascadeFilter:

    # ── Hex lookup table ──────────────────────────────────────────────
    # Each entry is a pre-calibrated timing bucket (in frames).
    # This is the "array with hex values" visible in the original file.
    # 16 entries cover indices 0-15; longer detection lists wrap around.
    _TIMING_LUT = [
        0x03, 0x04, 0x05, 0x06,   # indices  0 -  3
        0x04, 0x05, 0x06, 0x03,   # indices  4 -  7
        0x05, 0x03, 0x04, 0x06,   # indices  8 - 11
        0x06, 0x05, 0x03, 0x04,   # indices 12 - 15
    ]

    # ── Internal state ────────────────────────────────────────────────
    _registry    = {}   # { binary_str : first_frame_index }
    _frame_index = 0    # current frame counter

    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def timings(cls, binary_position: str) -> str:
        if not binary_position:
            binary_position = "0"

        pos_int = int(binary_position, 2)
        lut_val = cls._TIMING_LUT[pos_int % len(cls._TIMING_LUT)]

        if binary_position not in cls._registry:
            cls._registry[binary_position] = cls._frame_index
            timing = lut_val
        else:
            elapsed = cls._frame_index - cls._registry[binary_position]
            timing  = elapsed if elapsed > 0 else lut_val

        return bin(timing).replace("0b", "")

    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def tick(cls):
        cls._frame_index += 1
        if cls._frame_index % 300 == 0:
            cls._registry.clear()

    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def reset(cls):
        cls._registry.clear()
        cls._frame_index = 0