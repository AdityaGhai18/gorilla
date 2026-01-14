import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os
from pathlib import Path
import random

class BackgroundNoiseProcessor:
    def __init__(self, noise_dir="background_noise"):
        """
        Initialize the background noise processor.
        
        Args:
            noise_dir (str): Base directory containing the 'noise' subdirectory with noise files
        """
        self.noise_dir = noise_dir
        self._ensure_noise_directory()
        self.noise_files = self._load_noise_files()

    def _ensure_noise_directory(self):
        """Create the necessary directories if they don't exist."""
        Path(self.noise_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.noise_dir, "noise")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.noise_dir, "noisy_audio")).mkdir(parents=True, exist_ok=True)

    def _load_noise_files(self):
        """Load .wav, .webm, and .mp3 noise files from the noise directory."""
        noise_dir = os.path.join(self.noise_dir, "noise")
        noise_files = []
        # Look for .wav, .webm, and .mp3 files in the noise directory
        for ext in ("*.wav", "*.webm", "*.mp3"):
            found = list(Path(noise_dir).glob(ext))
            if found:
                noise_files.extend([str(f) for f in found])
        if noise_files:
            print(f"Found {len(noise_files)} noise files (.wav, .webm, .mp3)")
        else:
            print(f"Warning: No noise files found in {noise_dir}")
        return noise_files
    
    def add_background_noise(self, audio_path, noise_file, output_path=None, noise_level=-20):
        """
        Add background noise to an audio file.
        
        Args:
            audio_path (str): Path to the input audio file
            output_path (str): Path to save the output audio file (optional)
            noise_level (int): Volume of background noise in dB (default: -20)
            
        Returns:
            str: Path to the output audio file
        """
        # if not self.noise_files:
        #     raise ValueError("No background noise files found in the noise directory")
        if not os.path.exists(noise_file):
            raise FileNotFoundError(f"Noise file not found: {noise_file}")
        # Load the main audio
        audio = AudioSegment.from_file(audio_path)
        
        # # Randomly select a noise file
        # random.seed(16)
        # noise_file = random.choice(self.noise_files)
        noise = AudioSegment.from_file(noise_file)
        
        # Loop or trim noise to match speech duration
        target_duration = len(audio)
        if len(noise) < target_duration:
            # Loop the noise
            loops_needed = int(np.ceil(target_duration / len(noise)))
            noise = noise * loops_needed
        
        # Trim to exact length
        noise = noise[:target_duration]
        noise = noise + noise_level
        combined = audio.overlay(noise)
        
        if output_path is None:
            # Create noisy_audio directory inside background_noise
            noisy_dir = os.path.join(self.noise_dir, "noisy_audio")
            Path(noisy_dir).mkdir(parents=True, exist_ok=True)
            
            # Use just the filename without full path
            audio_filename = os.path.basename(audio_path)
            base_filename = os.path.splitext(audio_filename)[0]
            output_path = os.path.join(noisy_dir, f"{base_filename}_with_noise.mp3")
        
        combined.export(output_path, format="mp3")
        return output_path

    # def add_background_noise_batch(self, audio_dir, output_dir=None, noise_level=-20):
    #     """
    #     Process multiple audio files in a directory.
        
    #     Args:
    #         audio_dir (str): Directory containing input audio files
    #         output_dir (str): Directory to save processed files (optional)
    #         noise_level (int): Volume of background noise in dB (default: -20)
            
    #     Returns:
    #         list: Paths to all processed audio files
    #     """
    #     # If no output directory specified, use noisy_audio in background_noise directory
    #     if output_dir is None:
    #         output_dir = os.path.join(self.noise_dir, "noisy_audio")
        
    #     # Create output directory
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     processed_files = []
        
    #     # Get all wav files from input directory and its subdirectories
    #     audio_files = list(Path(audio_dir).rglob("*.wav"))
    #     total_files = len(audio_files)
        
    #     if total_files == 0:
    #         print(f"No .wav files found in {audio_dir}")
    #         return processed_files
        
    #     print(f"Processing {total_files} audio files...")
        
    #     # Process each audio file
    #     for i, audio_file in enumerate(audio_files, 1):
    #         # Preserve directory structure
    #         rel_path = audio_file.relative_to(Path(audio_dir))
    #         output_subdir = os.path.join(output_dir, os.path.dirname(str(rel_path)))
    #         Path(output_subdir).mkdir(parents=True, exist_ok=True)
            
    #         # Create output path
    #         output_path = os.path.join(output_subdir, f"{audio_file.stem}_with_noise.wav")
            
    #         try:
    #             processed_file = self.add_background_noise(
    #                 str(audio_file), output_path, noise_level
    #             )
    #             processed_files.append(processed_file)
    #             print(f"Processed {i}/{total_files}: {rel_path}")
    #         except Exception as e:
    #             print(f"Error processing {rel_path}: {str(e)}")
    #             continue
            
    #     print(f"\nCompleted processing {len(processed_files)} files")
    #     return processed_files

def overlay_audio_with_noise(speech_path, noise_path, output_path=None, noise_level_db=-20):
    """
    Overlay a speech audio file with another audio file as noise.
    
    Args:
        speech_path (str): Path to the speech audio file
        noise_path (str): Path to the noise audio file
        output_path (str): Path to save the output file (optional)
        noise_level_db (int): dB to reduce noise audio (default: -20)
        
    Returns:
        str: Path to the output file
    """
    # Load audio files
    speech = AudioSegment.from_file(speech_path)
    noise = AudioSegment.from_file(noise_path)

    # Loop or trim noise to match speech duration
    target_duration = len(speech)
    if len(noise) < target_duration:
        loops_needed = int(target_duration / len(noise)) + 1
        noise = noise * loops_needed
    noise = noise[:target_duration]

    # Adjust noise volume
    noise = noise + noise_level_db

    # Overlay
    combined = speech.overlay(noise)

    # Output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(speech_path))[0]
        noise_base = os.path.splitext(os.path.basename(noise_path))[0]
        output_path = f"{base}_with_{noise_base}_noise.mp3"

    combined.export(output_path, format="mp3")
    return output_path

def fluctuate_audio_volume(
    audio_path,
    output_path=None,
    min_db_change=-15,
    max_db_change=10,
    min_duration_ms=500,
    max_duration_ms=3000,
    n_fluctuations=5
):
    """
    Randomly fluctuate the volume of an audio clip, lowering and raising it at random intervals.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path to save the output audio file (optional)
        min_db_change (int): Minimum dB change (negative for lowering)
        max_db_change (int): Maximum dB change (positive for increasing)
        min_duration_ms (int): Minimum duration of a fluctuation in ms
        max_duration_ms (int): Maximum duration of a fluctuation in ms
        n_fluctuations (int): Number of fluctuations to apply
            
    Returns:
        str: Path to the output audio file
    """
    audio = AudioSegment.from_file(audio_path)
    length = len(audio)
    segments = []
    last_pos = 0
    flucts = []
    for _ in range(n_fluctuations):
        start = random.randint(0, length - min_duration_ms)
        dur = random.randint(min_duration_ms, min(max_duration_ms, length - start))
        db_change = random.uniform(min_db_change, max_db_change)
        flucts.append((start, start+dur, db_change))
    flucts.sort()  # sort by start time
    for start, end, db_change in flucts:
        if last_pos < start:
            segments.append(audio[last_pos:start])
        fluct_segment = audio[start:end] + db_change
        segments.append(fluct_segment)
        last_pos = end
    if last_pos < length:
        segments.append(audio[last_pos:])
    fluctuated = sum(segments)
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_fluctuated.mp3"
    fluctuated.export(output_path, format="mp3")
    print(f"Created fluctuated audio: {output_path}")
    return output_path

# def apply_gradual_noise_fade(
#     speech_path,
#     noise_path,
#     output_path=None,
#     min_db=-30,
#     max_db=0
# ):
#     """
#     Overlay noise on a speech file with a gradual fade out and in,
#     simulating someone walking away from and then back to a phone.
    
#     Args:
#         speech_path (str): Path to the speech audio file
#         noise_path (str): Path to the noise audio file
#         output_path (str): Path to save the output file (optional)
#         min_db (int): Minimum dB for noise (farthest point)
#         max_db (int): Maximum dB for noise (closest point)
        
#     Returns:
#         str: Path to the output file
#     """
#     from pydub import AudioSegment
#     import numpy as np
#     import os
    
#     speech = AudioSegment.from_file(speech_path)
#     noise = AudioSegment.from_file(noise_path)
#     duration = len(speech)
#     # Loop or trim noise to match speech duration
#     if len(noise) < duration:
#         loops_needed = int(np.ceil(duration / len(noise)))
#         noise = noise * loops_needed
#     noise = noise[:duration]
#     # Create a fade-out and fade-in 
#     half = duration // 2
#     envelope = np.concatenate([
#         np.linspace(max_db, min_db, half),  # fade out
#         np.linspace(min_db, max_db, duration - half)  # fade in
#     ])
#     # Apply envelope in chunks
#     chunk_ms = 100  # 0.1s
#     chunks = []
#     for i in range(0, duration, chunk_ms):
#         db = envelope[i] if i < len(envelope) else envelope[-1]
#         chunk = noise[i:i+chunk_ms] + db
#         chunks.append(chunk)
#     faded_noise = sum(chunks)
#     # Overlay
#     combined = speech.overlay(faded_noise)
#     if output_path is None:
#         base = os.path.splitext(os.path.basename(speech_path))[0]
#         noise_base = os.path.splitext(os.path.basename(noise_path))[0]
#         output_path = f"{base}_with_{noise_base}_walkaway.mp3"
#     combined.export(output_path, format="mp3")
#     print(f"Created: {output_path}")
#     return output_path

def apply_gradual_audio_fade(
    speech_path,
    output_path=None,
    min_db=-30,
    max_db=0
):
    """
    Gradually decrease and then increase the volume of the main audio (not the noise),
    simulating someone walking away from and then back to a phone.
    
    Args:
        speech_path (str): Path to the speech audio file
        output_path (str): Path to save the output file (optional)
        min_db (int): Minimum dB for audio (farthest point)
        max_db (int): Maximum dB for audio (closest point)
        
    Returns:
        str: Path to the output file
    """
    from pydub import AudioSegment
    import numpy as np
    import os
    
    speech = AudioSegment.from_file(speech_path)
    duration = len(speech)
    # Create a fade-out and fade-in
    half = duration // 2
    envelope = np.concatenate([
        np.linspace(max_db, min_db, half),  # fade out
        np.linspace(min_db, max_db, duration - half)  # fade in
    ])
    # Apply envelope in chunks
    chunk_ms = 100  # 0.1s
    segments = []
    for i in range(0, duration, chunk_ms):
        db = envelope[i] if i < len(envelope) else envelope[-1]
        chunk = speech[i:i+chunk_ms] + db
        segments.append(chunk)
    fluctuated = sum(segments)
    if output_path is None:
        base = os.path.splitext(os.path.basename(speech_path))[0]
        output_path = f"{base}_walkaway.mp3"
    fluctuated.export(output_path, format="mp3")
    print(f"Created: {output_path}")
    return output_path

def apply_network_cut_effect(
    audio_path,
    output_path=None,
    min_cut_ms=100,
    max_cut_ms=600,
    n_cuts=8
):
    """
    Simulate a network cut effect by muting random short segments in the audio.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path to save the output audio file (optional)
        min_cut_ms (int): Minimum duration of a cut in ms
        max_cut_ms (int): Maximum duration of a cut in ms
        n_cuts (int): Number of cuts to apply
            
    Returns:
        str: Path to the output audio file
    """
    from pydub import AudioSegment
    import random
    import os
    
    audio = AudioSegment.from_file(audio_path)
    length = len(audio)
    cuts = []
    for _ in range(n_cuts):
        start = random.randint(0, max(0, length - min_cut_ms))
        dur = random.randint(min_cut_ms, min(max_cut_ms, length - start))
        cuts.append((start, start+dur))
    cuts.sort()  # sort by start time
    segments = []
    last_pos = 0
    for start, end in cuts:
        if last_pos < start:
            segments.append(audio[last_pos:start])
        # Insert silence for the cut
        segments.append(AudioSegment.silent(duration=end-start, frame_rate=audio.frame_rate))
        last_pos = end
    if last_pos < length:
        segments.append(audio[last_pos:])
    glitched = sum(segments)
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_networkcut.mp3"
    glitched.export(output_path, format="mp3")
    print(f"Created: {output_path}")
    return output_path

def apply_network_beep_effect(
    audio_path,
    output_path=None,
    min_beep_ms=100,
    max_beep_ms=600,
    n_beeps=8,
    beep_freq=1000,
    beep_db=-10
):
    """
    Simulate a network cut effect by inserting beeps (instead of silence) at random short segments in the audio.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path to save the output audio file (optional)
        min_beep_ms (int): Minimum duration of a beep in ms
        max_beep_ms (int): Maximum duration of a beep in ms
        n_beeps (int): Number of beeps to insert
        beep_freq (int): Frequency of the beep in Hz
        beep_db (int): Volume of the beep in dB
            
    Returns:
        str: Path to the output audio file
    """
    from pydub import AudioSegment
    from pydub.generators import Sine
    import random
    import os
    
    audio = AudioSegment.from_file(audio_path)
    length = len(audio)
    beeps = []
    for _ in range(n_beeps):
        start = random.randint(0, max(0, length - min_beep_ms))
        dur = random.randint(min_beep_ms, min(max_beep_ms, length - start))
        beeps.append((start, start+dur))
    beeps.sort()  # sort by start time
    segments = []
    last_pos = 0
    for start, end in beeps:
        if last_pos < start:
            segments.append(audio[last_pos:start])
        # Insert beep for the cut
        beep = Sine(beep_freq).to_audio_segment(duration=end-start).apply_gain(beep_db)
        beep = beep.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
        segments.append(beep)
        last_pos = end
    if last_pos < length:
        segments.append(audio[last_pos:])
    glitched = sum(segments)
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_networkbeep.mp3"
    glitched.export(output_path, format="mp3")
    print(f"Created: {output_path}")
    return output_path

def apply_mic_rubbing_effect(audio_path, output_path=None):
    """
    Simulate a mic rubbing effect by overlaying a low-frequency rumble sound.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path to save the output audio file (optional)
    Returns:
        str: Path to the output audio file
    """
    from pydub import AudioSegment
    import os
    
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    
    #low-frequency rumble
    rumble_freq = 50  # Hz
    rumble = Sine(rumble_freq).to_audio_segment(duration=duration).apply_gain(-20)
    rumble = rumble.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
    
    # Overlay rumble on the original audio
    combined = audio.overlay(rumble)
    
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_micrubbing.mp3"

    combined.export(output_path, format="mp3")
    print(f"Created: {output_path}")
    return output_path

def apply_audio_mumbling_effect(audio_path, output_path=None):
    """
    Simulate a mumbling effect by applying a low-pass filter to the audio.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path to save the output audio file (optional)
    Returns:
        str: Path to the output audio file
    """
    from pydub import AudioSegment
    import os
    
    audio = AudioSegment.from_file(audio_path)
    
    #low-pass filter to simulate mumbling
    mumble = audio.low_pass_filter(1000)
    
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_mumbling.mp3"
    
    mumble.export(output_path, format="mp3")
    print(f"Created: {output_path}")
    return output_path

def apply_heavy_wind_effect(
    audio_path,
    wind_noise_path,
    output_path=None,
    noise_level_db=+5,
    n_hits=5,
    hit_min_freq=30,
    hit_max_freq=60,
    hit_min_duration_ms=80,
    hit_max_duration_ms=250,
    hit_db=+10
):
    """
    Simulates a heavy wind effect with background noise and mic buffeting.
    
    Args:
        audio_path (str): Path to the input audio file.
        wind_noise_path (str): Path to the wind noise audio file.
        output_path (str): Path to save the output file.
        noise_level_db (int): Volume of the constant wind noise.
        n_hits (int): Number of random mic hits to generate.
        hit_min_freq (int): Minimum frequency of a mic hit.
        hit_max_freq (int): Maximum frequency of a mic hit.
        hit_min_duration_ms (int): Minimum duration of a mic hit.
        hit_max_duration_ms (int): Maximum duration of a mic hit.
        hit_db (int): Volume of the mic hits relative to the audio.
    """
    from pydub import AudioSegment
    from pydub.generators import Sine
    import random
    import numpy as np

    # 1. Load audio and overlay the constant wind noise
    speech = AudioSegment.from_file(audio_path)
    wind_noise = AudioSegment.from_file(wind_noise_path)
    duration = len(speech)

    # Loop or trim wind noise to match speech duration
    if len(wind_noise) < duration:
        loops = int(np.ceil(duration / len(wind_noise)))
        wind_noise = wind_noise * loops
    wind_noise = wind_noise[:duration]

    # Combine speech with the base wind noise
    combined = speech.overlay(wind_noise + noise_level_db)

    # 2. Generate and overlay random mic hits (thumps)
    for _ in range(n_hits):
        # Create a short, low-frequency thump
        hit_freq = random.randint(hit_min_freq, hit_max_freq)
        hit_duration = random.randint(hit_min_duration_ms, hit_max_duration_ms)

        # quick attack, slightly longer decay
        thump = Sine(hit_freq).to_audio_segment(
            duration=hit_duration
        ).apply_gain(hit_db).fade_in(5).fade_out(hit_duration // 2)

        # Place the thump at a random position
        start_pos = random.randint(0, max(0, duration - hit_duration))
        
        # Overlay the thump onto the already noisy audio
        combined = combined.overlay(thump, position=start_pos)

    if output_path is None:
        base = Path(audio_path).stem
        output_path = f"{base}_wind.mp3"
    
    combined.export(output_path, format="mp3")
    print(f"Created: {output_path}")
    return output_path

def apply_reverb(
    audio_path,
    output_path=None,
    mode="cave",
    # echo params
    echo_delay_ms=120,
    echo_decay=0.6,
    echo_n=3,
    # cave params
    n_reflections=40,
    max_reflection_delay_ms=120,
    decay_mean=0.6,
    decay_std=0.12,
    lowpass_freq=3500,
    # convolution params
    ir_path=None
):
    """
    Unified reverb function.
    mode: "convolution" (requires ir_path), "echo", or "cave" (synthetic dense reverb).
    For "echo" mode uses echo_delay_ms, echo_decay, echo_n.
    For "cave" mode uses n_reflections, max_reflection_delay_ms, decay_mean, decay_std, lowpass_freq.
    For "convolution" mode provide ir_path.
    """
    import numpy as _np
    import random as _random
    import os as _os
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)

    if mode == "echo":
        combined = audio[:]  # copy
        for i in range(1, echo_n + 1):
            scale = echo_decay ** i
            gain_db = (-120.0 if scale <= 0 else 20.0 * _np.log10(scale))
            delayed_pos = int(echo_delay_ms * i)
            echo = (audio + gain_db)
            combined = combined.overlay(echo, position=delayed_pos)
        out = combined

    elif mode == "cave":
        duration = len(audio)
        max_tail = duration + max_reflection_delay_ms * 3
        reverb_container = AudioSegment.silent(duration=max_tail, frame_rate=audio.frame_rate)

        # early reflections
        for _ in range(n_reflections):
            delay = int(_random.random() ** 1.5 * max_reflection_delay_ms)
            decay = max(0.01, _np.random.normal(decay_mean, decay_std))
            gain_db = (-120.0 if decay <= 0 else 20.0 * _np.log10(decay))
            refl = (audio + gain_db)
            reverb_container = reverb_container.overlay(refl, position=delay)

        # long tail by appending fragments
        tail_fragment_len = 120
        n_tail_frags = int((max_tail - duration) / tail_fragment_len)
        for j in range(n_tail_frags):
            frag_start = int(_random.randint(0, max(0, duration - tail_fragment_len)))
            frag = audio[frag_start:frag_start + tail_fragment_len]
            frac = (j + 1) / max(1, n_tail_frags)
            this_decay = max(0.02, decay_mean * (1.0 - frac) ** 1.5)
            gain_db = (-120.0 if this_decay <= 0 else 20.0 * _np.log10(this_decay))
            pos = duration + j * tail_fragment_len
            reverb_container = reverb_container.overlay(frag + gain_db, position=pos)

        try:
            reverb_container = reverb_container.low_pass_filter(lowpass_freq)
        except Exception:
            pass

        wet_level_db = -3
        wet = reverb_container[:len(audio) + max_reflection_delay_ms]
        out = audio.overlay(wet + wet_level_db)

    elif mode == "convolution":
        if not ir_path:
            raise ValueError("ir_path is required for convolution mode")
        # convert to mono and match sample rate
        speech = audio.set_channels(1)
        ir = AudioSegment.from_file(ir_path).set_channels(1)
        if ir.frame_rate != speech.frame_rate:
            ir = ir.set_frame_rate(speech.frame_rate)
        sr = speech.frame_rate
        sw = speech.sample_width

        s_arr = _np.array(speech.get_array_of_samples()).astype(_np.float32)
        ir_arr = _np.array(ir.get_array_of_samples()).astype(_np.float32)
        if s_arr.size == 0 or ir_arr.size == 0:
            raise ValueError("Empty audio or IR file.")

        conv = _np.convolve(s_arr, ir_arr)
        max_abs = _np.max(_np.abs(conv))
        if max_abs == 0:
            max_abs = 1.0
        max_int = float(2 ** (8 * sw - 1) - 1)
        conv_norm = (conv / max_abs) * (0.9 * max_int)

        if sw == 2:
            conv_int = conv_norm.astype(_np.int16)
        elif sw == 4:
            conv_int = conv_norm.astype(_np.int32)
        else:
            conv_int = conv_norm.astype(_np.int16)

        out = AudioSegment(
            conv_int.tobytes(),
            frame_rate=sr,
            sample_width=sw,
            channels=1
        )

    else:
        raise ValueError(f"Unknown reverb mode: {mode}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_reverb_{mode}.mp3" if mode != "convolution" else f"{base}_reverb_conv.mp3"
    out.export(output_path, format="mp3")
    print(f"Created reverb ({mode}): {output_path}")
    return output_path

def apply_clipping_distortion(audio_path, output_path=None, clip_threshold=0.8, drive_db=0.0):
    """
    Simulate microphone overload: apply gain (drive_db) then non-linear clipping at clip_threshold (0..1 scale).
    clip_threshold close to 1.0 is mild, lower values produce heavy clipping.
    """
    import numpy as _np
    import os as _os
    from pydub import AudioSegment

    seg = AudioSegment.from_file(audio_path)
    sr = seg.frame_rate
    sw = seg.sample_width
    channels = seg.channels

    # apply drive
    seg = seg.apply_gain(drive_db)

    arr = _np.array(seg.get_array_of_samples()).astype(_np.float32)
    # interleaved if stereo; normalize by max int based on sample width
    max_int = float(2 ** (8 * sw - 1) - 1)
    arr = arr / max_int  # now in [-1,1]

    # apply soft clipping (tanh) followed by hard clip to threshold
    soft = _np.tanh(arr * 3.0)  # pre-emphasize harmonics
    clipped = _np.clip(soft, -clip_threshold, clip_threshold)

    # scale back
    out_arr = (clipped * max_int * 0.98).astype(_np.int16 if sw == 2 else _np.int32)

    out_seg = AudioSegment(out_arr.tobytes(), frame_rate=sr, sample_width=sw, channels=channels)
    if output_path is None:
        base = _os.path.splitext(_os.path.basename(audio_path))[0]
        output_path = f"{base}_clipped.mp3"
    out_seg.export(output_path, format="mp3")
    print(f"Created clipped/distorted audio: {output_path}")
    return output_path

def apply_gain_variation(
    audio_path,
    output_path=None,
    min_gain_db=-12,
    max_gain_db=6,
    segment_ms=400
):
    """
    Randomly vary input gain over time to simulate moving mic distance or user loudness changes.
    """
    import random as _random
    import os as _os
    from pydub import AudioSegment

    seg = AudioSegment.from_file(audio_path)
    duration = len(seg)
    out_segments = []
    for pos in range(0, duration, segment_ms):
        chunk = seg[pos:pos + segment_ms]
        gain = _random.uniform(min_gain_db, max_gain_db)
        out_segments.append(chunk.apply_gain(gain))
    out = sum(out_segments)
    if output_path is None:
        base = _os.path.splitext(_os.path.basename(audio_path))[0]
        output_path = f"{base}_gainvar.mp3"
    out.export(output_path, format="mp3")
    print(f"Created gain-varied audio: {output_path}")
    return output_path

def apply_mechanical_interference(
    audio_path,
    output_path=None,
    n_rubs=3,
    rub_freq_range=(40, 120),
    rub_db=-6,
    n_clicks=8,
    click_db=0
):
    """
    Overlay mechanical noises: low-frequency rubbing/thumps and short clicks/squeaks.
    """
    from pydub import AudioSegment
    from pydub.generators import Sine
    import random
    import os

    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    combined = audio[:]  # base

    # low-frequency rubs/thumps
    for _ in range(n_rubs):
        freq = random.randint(rub_freq_range[0], rub_freq_range[1])
        dur = random.randint(150, 700)
        start = random.randint(0, max(0, duration - dur))
        rumble = Sine(freq).to_audio_segment(duration=dur).apply_gain(rub_db).fade_in(20).fade_out(100)
        rumble = rumble.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
        combined = combined.overlay(rumble, position=start)

    # clicks/squeaks: very short high-gain narrow pulses
    for _ in range(n_clicks):
        click_dur = random.randint(6, 30)
        freq = random.choice([800, 1600, 2400, 3200])
        pos = random.randint(0, max(0, duration - click_dur))
        click = Sine(freq).to_audio_segment(duration=click_dur).apply_gain(click_db).fade_out(10)
        click = click.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
        combined = combined.overlay(click, position=pos)

    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_mechint.mp3"
    combined.export(output_path, format="mp3")
    print(f"Created mechanical interference audio: {output_path}")
    return output_path