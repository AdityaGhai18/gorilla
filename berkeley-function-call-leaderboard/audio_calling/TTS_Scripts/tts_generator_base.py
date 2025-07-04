import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

class TTSGeneratorBase(ABC):
    """Abstract base class for TTS generators."""
    
    def __init__(self, provider_name: str, output_root: Optional[str] = None):
        self.provider_name = provider_name
        if output_root is not None:
            self.output_root = Path(output_root)
        else:
            # Default to audio_calling/audio/provider_name for output
            self.output_root = Path(__file__).parent.parent / 'audio' / provider_name
        self.folder_map = {
            'simple': self.output_root / 'simple',
            'multiple': self.output_root / 'multiple',
            'multi_turn': self.output_root / 'multi_turn',
        }
        self.setup_output_directories()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the TTS client/API. To be implemented by each provider."""
        pass
    
    @abstractmethod
    def _generate_audio(self, text: str) -> bytes:
        """Generate audio from text. To be implemented by each provider."""
        pass
    
    def setup_output_directories(self):
        """Create output directories for each format."""
        for folder in self.folder_map.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def load_json_data(self, file_path: str) -> List[dict]:
        """Load and parse JSON data from file."""
        # Try relative to audio_calling directory first (for latest_results/)
        input_path = Path(__file__).parent.parent / file_path
        if not input_path.exists():
            # If not found, try relative to current file
            input_path = Path(__file__).parent / file_path
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def detect_format(self, data: List[dict]) -> str:
        """Auto-detect if data is multi_turn, simple, or multiple format."""
        if not data or 'question' not in data[0]:
            raise ValueError("Invalid data format: missing 'question' field")
        
        first_question = data[0]['question']
        if isinstance(first_question, list) and len(first_question) > 1:
            return 'multi_turn'
        elif isinstance(first_question, list) and len(first_question) == 1:
            return 'simple'  # Will be refined to 'multiple' if filename contains 'multiple'
        else:
            raise ValueError("Could not determine data format")
    
    def extract_transformed_content(self, data: List[dict], test_index: int, turn_index: int = 0) -> str:
        """Extract transformed_content from JSON structure."""
        if test_index >= len(data):
            raise IndexError(f"Test index {test_index} out of range")
        
        test_case = data[test_index]
        if 'question' not in test_case:
            raise ValueError(f"Test case {test_index} missing 'question' field")
        
        question = test_case['question']
        if not isinstance(question, list) or len(question) <= turn_index:
            raise IndexError(f"Turn index {turn_index} out of range")
        
        turn = question[turn_index]
        if not isinstance(turn, list) or len(turn) == 0:
            raise ValueError(f"Invalid turn structure at index {turn_index}")
        
        utterance = turn[0]
        if 'transformed_content' not in utterance:
            raise ValueError(f"Missing 'transformed_content' in utterance")
        
        return utterance['transformed_content']
    
    def save_audio(self, audio_bytes: bytes, file_path: Path):
        """Save audio bytes to file."""
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"Saved: {file_path}")
    
    def process_test_cases(self, data: List[dict], num_cases: int = 1, input_filename: str = ""):
        """Process test cases and generate audio files."""
        format_type = self.detect_format(data)
        
        # Refine format detection for simple vs multiple
        if format_type == 'simple' and 'multiple' in input_filename:
            format_type = 'multiple'
        
        print(f"Detected format: {format_type}")
        output_dir = self.folder_map[format_type]
        
        # Determine file extension based on provider
        file_extension = "mp3" if self.provider_name == "cartesia" else "wav"
        
        for i in range(min(num_cases, len(data))):
            test_id = data[i].get('id', f'test_case_{i}')
            
            try:
                if format_type == 'multi_turn':
                    # Process ALL turns for this test case before moving to the next
                    question = data[i]['question']
                    for turn_idx in range(len(question)):
                        try:
                            transformed_content = self.extract_transformed_content(data, i, turn_idx)
                            audio_filename = f"{test_id}_turn{turn_idx+1}.{file_extension}"
                            
                            if not transformed_content:
                                print(f"No transformed_content found in test case {i+1}, turn {turn_idx+1}.")
                                continue
                            
                            audio_path = output_dir / audio_filename
                            
                            # Generate audio using provider-specific method
                            audio_bytes = self._generate_audio(transformed_content)
                            
                            # Save audio file
                            self.save_audio(audio_bytes, audio_path)
                            
                        except Exception as e:
                            print(f"Error processing test case {i+1}, turn {turn_idx+1}: {e}")
                else:
                    # Process single turn for simple/multiple
                    transformed_content = self.extract_transformed_content(data, i, 0)
                    audio_filename = f"{test_id}.{file_extension}"
                    
                    if not transformed_content:
                        print(f"No transformed_content found in test case {i+1}.")
                        continue
                    
                    audio_path = output_dir / audio_filename
                    
                    # Generate audio using provider-specific method
                    audio_bytes = self._generate_audio(transformed_content)
                    
                    # Save audio file
                    self.save_audio(audio_bytes, audio_path)
                
            except Exception as e:
                print(f"Error processing test case {i+1}: {e}")
    
    def run(self, input_file_path: str, num_cases: int = 1):
        """Main execution method."""
        try:
            # Load data
            data = self.load_json_data(input_file_path)
            print(f"Loaded {len(data)} test cases from {input_file_path}")
            
            # Process test cases
            self.process_test_cases(data, num_cases, input_file_path)
            
        except Exception as e:
            print(f"Error: {e}") 