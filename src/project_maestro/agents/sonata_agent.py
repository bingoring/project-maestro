"""Sonata Agent - specialized in generating game audio (music and sound effects)."""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel

from ..core.agent_framework import BaseAgent, AgentType, AgentTask
from ..core.message_queue import EventType, publish_event
from ..core.config import settings


class AudioRequest(BaseModel):
    """Request for audio generation."""
    project_id: str = Field(description="Project identifier")
    audio_type: str = Field(description="Type of audio (bgm, sfx)")
    style: str = Field(description="Music style or sound description")
    duration: float = Field(description="Audio duration in seconds")
    mood: str = Field(description="Mood or atmosphere")
    instruments: List[str] = Field(default=[], description="Preferred instruments")


class GeneratedAudio(BaseModel):
    """Generated audio asset."""
    filename: str = Field(description="Audio filename")
    audio_type: str = Field(description="Type of audio")
    duration: float = Field(description="Audio duration in seconds")
    sample_rate: int = Field(description="Sample rate in Hz")
    file_size: int = Field(description="File size in bytes")
    description: str = Field(description="Audio description")
    generation_params: Dict[str, Any] = Field(description="Generation parameters used")


class SunoAPI:
    """Interface for Suno AI music generation API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.suno_api_key
        self.base_url = "https://api.suno.ai/v1"
        
    async def generate_music(
        self,
        prompt: str,
        duration: float = 30.0,
        style: str = "instrumental",
        mood: str = "upbeat"
    ) -> bytes:
        """Generate music using Suno API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "duration": duration,
            "style": style,
            "mood": mood,
            "instrumental": True,
            "format": "wav"
        }
        
        # Note: This is a mock implementation as Suno API details may vary
        # In a real implementation, you would use the actual Suno API endpoints
        response = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Suno API Error: {response.status_code} - {response.text}")


class AudioProcessor:
    """Utility class for audio processing and optimization."""
    
    @staticmethod
    def normalize_audio(input_file: Path, output_file: Path) -> None:
        """Normalize audio levels using ffmpeg."""
        try:
            subprocess.run([
                "ffmpeg", "-i", str(input_file),
                "-filter:a", "loudnorm",
                "-y", str(output_file)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Audio normalization failed: {e}")
            
    @staticmethod
    def convert_to_ogg(input_file: Path, output_file: Path, quality: int = 5) -> None:
        """Convert audio to OGG format for better compression."""
        try:
            subprocess.run([
                "ffmpeg", "-i", str(input_file),
                "-c:a", "libvorbis", "-q:a", str(quality),
                "-y", str(output_file)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Audio conversion failed: {e}")
            
    @staticmethod
    def trim_silence(input_file: Path, output_file: Path) -> None:
        """Remove silence from beginning and end of audio."""
        try:
            subprocess.run([
                "ffmpeg", "-i", str(input_file),
                "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB",
                "-y", str(output_file)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Silence removal failed: {e}")
            
    @staticmethod
    def create_loop(input_file: Path, output_file: Path, loop_duration: float) -> None:
        """Create a seamless loop from audio."""
        try:
            subprocess.run([
                "ffmpeg", "-i", str(input_file),
                "-filter:a", f"afade=t=out:st={loop_duration-1}:d=1",
                "-t", str(loop_duration),
                "-y", str(output_file)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Loop creation failed: {e}")


class SoundEffectGenerator:
    """Generator for procedural sound effects."""
    
    @staticmethod
    async def generate_jump_sound() -> bytes:
        """Generate a jump sound effect."""
        # This would use a procedural audio generation library
        # For now, we'll create a simple sine wave burst
        return SoundEffectGenerator._create_sine_burst(
            frequency=440, duration=0.2, amplitude=0.5
        )
        
    @staticmethod
    async def generate_collect_sound() -> bytes:
        """Generate a collectible pickup sound."""
        return SoundEffectGenerator._create_chime(
            frequencies=[523, 659, 784], duration=0.3
        )
        
    @staticmethod
    async def generate_damage_sound() -> bytes:
        """Generate a damage/hit sound effect."""
        return SoundEffectGenerator._create_noise_burst(
            duration=0.1, amplitude=0.3
        )
        
    @staticmethod
    def _create_sine_burst(frequency: float, duration: float, amplitude: float) -> bytes:
        """Create a simple sine wave burst."""
        import numpy as np
        import wave
        import io
        
        sample_rate = 44100
        samples = int(sample_rate * duration)
        
        # Generate sine wave
        t = np.linspace(0, duration, samples)
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        envelope = np.exp(-t * 5)  # Exponential decay
        audio *= envelope
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)
        
        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
            
        return buffer.getvalue()
        
    @staticmethod
    def _create_chime(frequencies: List[float], duration: float) -> bytes:
        """Create a chime sound with multiple frequencies."""
        import numpy as np
        import wave
        import io
        
        sample_rate = 44100
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        audio = np.zeros(samples)
        for freq in frequencies:
            component = 0.3 * np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 3)
            audio += component * envelope
            
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767 * 0.5).astype(np.int16)
        
        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
            
        return buffer.getvalue()
        
    @staticmethod
    def _create_noise_burst(duration: float, amplitude: float) -> bytes:
        """Create a noise burst for damage sounds."""
        import numpy as np
        import wave
        import io
        
        sample_rate = 44100
        samples = int(sample_rate * duration)
        
        # Generate filtered noise
        noise = np.random.normal(0, amplitude, samples)
        
        # Apply low-pass filter (simple moving average)
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        filtered_noise = np.convolve(noise, kernel, mode='same')
        
        # Apply envelope
        t = np.linspace(0, duration, samples)
        envelope = np.exp(-t * 20)  # Fast decay
        audio = filtered_noise * envelope
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)
        
        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
            
        return buffer.getvalue()


class SonataTools:
    """Tools available to the Sonata Agent."""
    
    @staticmethod
    def create_generate_bgm_tool(agent: "SonataAgent") -> BaseTool:
        """Tool for generating background music."""
        
        class BGMInput(LangchainBaseModel):
            mood: str = Field(description="Music mood (upbeat, calm, tense, mysterious)")
            genre: str = Field(description="Music genre (orchestral, electronic, rock, ambient)")
            duration: float = Field(description="Duration in seconds")
            loop: bool = Field(default=True, description="Should be loopable")
            
        async def generate_bgm(
            mood: str,
            genre: str,
            duration: float,
            loop: bool
        ) -> str:
            try:
                audio = await agent._generate_background_music(
                    mood, genre, duration, loop
                )
                return f"Generated {duration}s {mood} {genre} BGM"
            except Exception as e:
                return f"Error generating BGM: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_bgm,
            name="generate_bgm",
            description="Generate background music",
            args_schema=BGMInput
        )
        
    @staticmethod
    def create_generate_sfx_tool(agent: "SonataAgent") -> BaseTool:
        """Tool for generating sound effects."""
        
        class SFXInput(LangchainBaseModel):
            effect_type: str = Field(description="Type of sound effect (jump, collect, damage, shoot)")
            intensity: str = Field(description="Sound intensity (soft, medium, loud)")
            pitch: str = Field(description="Sound pitch (low, medium, high)")
            
        async def generate_sfx(
            effect_type: str,
            intensity: str,
            pitch: str
        ) -> str:
            try:
                audio = await agent._generate_sound_effect(
                    effect_type, intensity, pitch
                )
                return f"Generated {effect_type} sound effect ({intensity}, {pitch})"
            except Exception as e:
                return f"Error generating SFX: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_sfx,
            name="generate_sfx",
            description="Generate sound effects",
            args_schema=SFXInput
        )
        
    @staticmethod
    def create_generate_ambient_tool(agent: "SonataAgent") -> BaseTool:
        """Tool for generating ambient sounds."""
        
        class AmbientInput(LangchainBaseModel):
            environment: str = Field(description="Environment type (forest, city, dungeon, space)")
            intensity: float = Field(description="Ambient intensity (0.0 to 1.0)")
            duration: float = Field(description="Duration in seconds")
            
        async def generate_ambient(
            environment: str,
            intensity: float,
            duration: float
        ) -> str:
            try:
                audio = await agent._generate_ambient_sound(
                    environment, intensity, duration
                )
                return f"Generated {duration}s {environment} ambient sound"
            except Exception as e:
                return f"Error generating ambient sound: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_ambient,
            name="generate_ambient",
            description="Generate ambient environment sounds",
            args_schema=AmbientInput
        )


class SonataAgent(BaseAgent):
    """Specialist agent for generating game audio assets."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="sonata_agent",
            agent_type=AgentType.SONATA,
            **kwargs
        )
        
        self.suno_api = SunoAPI() if settings.suno_api_key else None
        self.processor = AudioProcessor()
        self.sfx_generator = SoundEffectGenerator()
        self.generated_audio: Dict[str, GeneratedAudio] = {}
        
        # Create tools
        self.tools = [
            SonataTools.create_generate_bgm_tool(self),
            SonataTools.create_generate_sfx_tool(self),
            SonataTools.create_generate_ambient_tool(self),
        ]
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for audio generation."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Sonata agent."""
        return """
        You are the Sonata Agent, a specialist in generating high-quality game audio assets.
        
        Your expertise includes:
        - Background music composition in various genres
        - Sound effect design and creation
        - Ambient soundscape generation
        - Audio optimization for mobile games
        - Music theory and sound design principles
        
        Audio generation capabilities:
        - BGM: Orchestral, electronic, rock, ambient, chiptune styles
        - SFX: UI sounds, gameplay effects, character sounds
        - Ambient: Environmental soundscapes and atmospheres
        - Adaptive audio: Dynamic and interactive music systems
        
        Technical considerations:
        - Mobile-optimized file formats (OGG Vorbis)
        - Appropriate compression levels
        - Seamless loops for background music
        - Balanced audio levels and dynamics
        - Memory-efficient audio streaming
        
        Quality standards:
        - Professional audio quality
        - Consistent volume levels
        - Clean, artifact-free sound
        - Appropriate dynamic range
        - Genre-appropriate instrumentation
        - Emotional coherence with game content
        
        Always consider the game's mood, target audience, and technical constraints
        when generating audio assets.
        """
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute an audio generation task."""
        
        action = task.action
        params = task.parameters
        
        if action == "generate_sound":
            return await self._handle_sound_generation(params)
        elif action == "generate_music":
            return await self._handle_music_generation(params)
        elif action == "generate_ambient":
            return await self._handle_ambient_generation(params)
        else:
            # Use agent executor for complex audio generation
            result = await self.agent_executor.ainvoke({
                "input": f"Generate audio for: {action} with specifications: {params}"
            })
            return {"result": result["output"]}
            
    async def _handle_sound_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sound effect generation."""
        
        sound = params.get("sound", {})
        project_id = params.get("project_id", "")
        
        sound_type = sound.get("type", "sfx")
        name = sound.get("name", "sound")
        description = sound.get("description", "game sound")
        
        self.log_agent_action("generate_sound", "started",
                            sound=name, project_id=project_id)
        
        try:
            if sound_type == "bgm":
                audio = await self._generate_background_music(
                    mood=sound.get("mood", "neutral"),
                    genre=sound.get("genre", "orchestral"),
                    duration=sound.get("duration", 30.0),
                    loop=True
                )
            else:
                audio = await self._generate_sound_effect(
                    effect_type=name,
                    intensity=sound.get("intensity", "medium"),
                    pitch=sound.get("pitch", "medium")
                )
                
            # Store audio
            asset_key = f"{project_id}_{name}_audio"
            self.generated_audio[asset_key] = audio
            
            # Publish asset generated event
            await publish_event(
                EventType.ASSET_GENERATED,
                "sonata_agent",
                {
                    "asset_type": "audio",
                    "filename": audio.filename,
                    "sound": name,
                    "project_id": project_id,
                    "audio_type": sound_type
                }
            )
            
            return {
                "filename": audio.filename,
                "audio_type": audio.audio_type,
                "duration": audio.duration,
                "file_size": audio.file_size
            }
            
        except Exception as e:
            self.log_agent_error("generate_sound", e,
                               sound=name, project_id=project_id)
            raise
            
    async def _generate_background_music(
        self,
        mood: str,
        genre: str,
        duration: float,
        loop: bool
    ) -> GeneratedAudio:
        """Generate background music."""
        
        if self.suno_api:
            # Use Suno API for music generation
            prompt = f"{genre} style background music, {mood} mood, instrumental, game music"
            
            try:
                audio_data = await self.suno_api.generate_music(
                    prompt=prompt,
                    duration=duration,
                    style=genre,
                    mood=mood
                )
                
                # Save raw audio
                filename = f"{mood}_{genre}_bgm.wav"
                raw_path = settings.data_dir / "temp" / filename
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(raw_path, 'wb') as f:
                    f.write(audio_data)
                    
            except Exception as e:
                self.logger.warning(f"Suno API failed, using procedural generation: {e}")
                audio_data = await self._generate_procedural_music(mood, genre, duration)
                
                filename = f"{mood}_{genre}_bgm_proc.wav"
                raw_path = settings.data_dir / "temp" / filename
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(raw_path, 'wb') as f:
                    f.write(audio_data)
        else:
            # Use procedural generation
            audio_data = await self._generate_procedural_music(mood, genre, duration)
            
            filename = f"{mood}_{genre}_bgm_proc.wav"
            raw_path = settings.data_dir / "temp" / filename
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(raw_path, 'wb') as f:
                f.write(audio_data)
                
        # Process audio
        processed_filename = filename.replace('.wav', '_processed.ogg')
        processed_path = settings.data_dir / "assets" / "audio" / "bgm" / processed_filename
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize and convert to OGG
        normalized_path = raw_path.with_suffix('.normalized.wav')
        self.processor.normalize_audio(raw_path, normalized_path)
        
        if loop:
            loop_path = raw_path.with_suffix('.loop.wav')
            self.processor.create_loop(normalized_path, loop_path, duration)
            self.processor.convert_to_ogg(loop_path, processed_path)
        else:
            self.processor.convert_to_ogg(normalized_path, processed_path)
            
        # Clean up temp files
        raw_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)
        if loop:
            loop_path.unlink(missing_ok=True)
            
        return GeneratedAudio(
            filename=processed_filename,
            audio_type="bgm",
            duration=duration,
            sample_rate=44100,
            file_size=processed_path.stat().st_size,
            description=f"{mood} {genre} background music",
            generation_params={
                "mood": mood,
                "genre": genre,
                "duration": duration,
                "loop": loop
            }
        )
        
    async def _generate_sound_effect(
        self,
        effect_type: str,
        intensity: str,
        pitch: str
    ) -> GeneratedAudio:
        """Generate sound effects."""
        
        # Use procedural SFX generation
        if effect_type.lower() in ["jump", "hop"]:
            audio_data = await self.sfx_generator.generate_jump_sound()
        elif effect_type.lower() in ["collect", "pickup", "coin"]:
            audio_data = await self.sfx_generator.generate_collect_sound()
        elif effect_type.lower() in ["damage", "hit", "hurt"]:
            audio_data = await self.sfx_generator.generate_damage_sound()
        else:
            # Default to generic sound
            audio_data = await self.sfx_generator.generate_collect_sound()
            
        # Save and process
        filename = f"{effect_type}_{intensity}_{pitch}.wav"
        raw_path = settings.data_dir / "temp" / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(raw_path, 'wb') as f:
            f.write(audio_data)
            
        # Process and convert
        processed_filename = filename.replace('.wav', '.ogg')
        processed_path = settings.data_dir / "assets" / "audio" / "sfx" / processed_filename
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Trim silence and convert
        trimmed_path = raw_path.with_suffix('.trimmed.wav')
        self.processor.trim_silence(raw_path, trimmed_path)
        self.processor.convert_to_ogg(trimmed_path, processed_path, quality=6)
        
        # Get duration
        duration = await self._get_audio_duration(processed_path)
        
        # Clean up
        raw_path.unlink(missing_ok=True)
        trimmed_path.unlink(missing_ok=True)
        
        return GeneratedAudio(
            filename=processed_filename,
            audio_type="sfx",
            duration=duration,
            sample_rate=44100,
            file_size=processed_path.stat().st_size,
            description=f"{effect_type} sound effect ({intensity}, {pitch})",
            generation_params={
                "effect_type": effect_type,
                "intensity": intensity,
                "pitch": pitch
            }
        )
        
    async def _generate_procedural_music(
        self,
        mood: str,
        genre: str,
        duration: float
    ) -> bytes:
        """Generate music using procedural methods."""
        # This is a simplified procedural music generator
        # In a real implementation, you might use libraries like:
        # - mingus for music theory
        # - pydub for audio manipulation
        # - numpy for signal generation
        
        import numpy as np
        import wave
        import io
        
        sample_rate = 44100
        samples = int(sample_rate * duration)
        
        # Generate a simple chord progression
        if mood == "happy":
            frequencies = [261.63, 329.63, 392.00, 523.25]  # C major chord
        elif mood == "sad":
            frequencies = [220.00, 261.63, 311.13]  # A minor chord
        elif mood == "tense":
            frequencies = [233.08, 277.18, 329.63]  # Bb diminished
        else:
            frequencies = [261.63, 329.63, 392.00]  # Default C major
            
        t = np.linspace(0, duration, samples)
        audio = np.zeros(samples)
        
        # Generate chord tones
        for i, freq in enumerate(frequencies):
            amplitude = 0.2 / len(frequencies)
            wave_component = amplitude * np.sin(2 * np.pi * freq * t)
            
            # Add some variation based on genre
            if genre == "electronic":
                wave_component *= (1 + 0.3 * np.sin(2 * np.pi * freq * 2 * t))
            elif genre == "orchestral":
                wave_component *= np.exp(-t * 0.1)  # Natural decay
                
            audio += wave_component
            
        # Add rhythm pattern
        beat_freq = 2.0  # 2 beats per second
        beat_pattern = 0.5 * (1 + np.sin(2 * np.pi * beat_freq * t))
        audio *= beat_pattern
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)
        
        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
            
        return buffer.getvalue()
        
    async def _get_audio_duration(self, file_path: Path) -> float:
        """Get audio duration using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-i", str(file_path),
                "-show_entries", "format=duration",
                "-v", "quiet", "-of", "csv=p=0"
            ], capture_output=True, text=True)
            
            return float(result.stdout.strip())
        except Exception:
            return 0.0
            
    async def _handle_music_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle music generation requests."""
        mood = params.get("mood", "neutral")
        genre = params.get("genre", "orchestral")
        duration = params.get("duration", 30.0)
        loop = params.get("loop", True)
        
        audio = await self._generate_background_music(mood, genre, duration, loop)
        
        return {
            "filename": audio.filename,
            "duration": audio.duration,
            "file_size": audio.file_size
        }
        
    async def _handle_ambient_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ambient sound generation."""
        environment = params.get("environment", "forest")
        intensity = params.get("intensity", 0.5)
        duration = params.get("duration", 60.0)
        
        audio = await self._generate_ambient_sound(environment, intensity, duration)
        
        return {
            "filename": audio.filename,
            "duration": audio.duration,
            "file_size": audio.file_size
        }
        
    async def _generate_ambient_sound(
        self,
        environment: str,
        intensity: float,
        duration: float
    ) -> GeneratedAudio:
        """Generate ambient environmental sounds."""
        # Simplified ambient sound generation
        audio_data = await self._generate_procedural_ambient(environment, intensity, duration)
        
        filename = f"{environment}_ambient.wav"
        raw_path = settings.data_dir / "temp" / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(raw_path, 'wb') as f:
            f.write(audio_data)
            
        # Process
        processed_filename = filename.replace('.wav', '.ogg')
        processed_path = settings.data_dir / "assets" / "audio" / "ambient" / processed_filename
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.processor.convert_to_ogg(raw_path, processed_path, quality=4)
        raw_path.unlink(missing_ok=True)
        
        return GeneratedAudio(
            filename=processed_filename,
            audio_type="ambient",
            duration=duration,
            sample_rate=44100,
            file_size=processed_path.stat().st_size,
            description=f"{environment} ambient sound",
            generation_params={
                "environment": environment,
                "intensity": intensity,
                "duration": duration
            }
        )
        
    async def _generate_procedural_ambient(
        self,
        environment: str,
        intensity: float,
        duration: float
    ) -> bytes:
        """Generate procedural ambient sounds."""
        import numpy as np
        import wave
        import io
        
        sample_rate = 44100
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Base noise
        noise = np.random.normal(0, 0.1, samples) * intensity
        
        # Add environment-specific characteristics
        if environment == "forest":
            # Add bird chirps and wind
            for _ in range(int(duration * 0.5)):  # Random chirps
                chirp_start = np.random.randint(0, samples - 1000)
                chirp_freq = np.random.uniform(800, 2000)
                chirp_duration = 0.3
                chirp_samples = int(sample_rate * chirp_duration)
                
                if chirp_start + chirp_samples < samples:
                    chirp_t = np.linspace(0, chirp_duration, chirp_samples)
                    chirp = 0.1 * np.sin(2 * np.pi * chirp_freq * chirp_t)
                    chirp *= np.exp(-chirp_t * 5)  # Decay
                    noise[chirp_start:chirp_start + chirp_samples] += chirp
                    
        elif environment == "city":
            # Add distant traffic hum
            traffic_freq = 60  # Low frequency hum
            traffic = 0.05 * np.sin(2 * np.pi * traffic_freq * t) * intensity
            noise += traffic
            
        # Apply low-pass filter
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 0.1, btype='low')
        filtered_noise = filtfilt(b, a, noise)
        
        # Normalize
        audio = filtered_noise / np.max(np.abs(filtered_noise)) * 0.6
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)
        
        # Create WAV
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
            
        return buffer.getvalue()