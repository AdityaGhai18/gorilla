import json
import sys
import os
import argparse
import time
import pickle
import signal
import atexit
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from audio_calling.clean_to_speech_text.granular_speech_pipeline import GranularSpeechPipeline, PipelineConfig
from audio_calling.clean_to_speech_text.clarification_extractor import ClarificationExtractor

# Maximum concurrency configuration
MAX_WORKERS = 250  # All workers running simultaneously
CASES_PER_WORKER = 1  # Each worker processes exactly 1 case at a time

# BULLETPROOF settings
CHECKPOINT_INTERVAL = 100  # Save progress every 100 cases
MAX_RETRIES = 3  # Retry failed API calls
RETRY_DELAY = 1.0  # Delay between retries
FORCE_SAVE_INTERVAL = 300  # Force save every 5 minutes

# Global variables for checkpointing
processed_cases = {}
failed_cases = []
start_time = None
checkpoint_file = None
output_file = None
data_to_process = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('max_concurrency_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle signals gracefully and save progress."""
    logger.warning(f"Received signal {signum}, saving progress and exiting gracefully...")
    save_checkpoint()
    save_output_files()
    sys.exit(0)

def save_checkpoint():
    """Save progress checkpoint - called on exit and periodically."""
    global processed_cases, failed_cases, data_to_process
    
    if not data_to_process:
        return
        
    try:
        checkpoint_data = {
            'processed_cases': processed_cases,
            'failed_cases': failed_cases,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(data_to_process)
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        logger.info(f"✅ Checkpoint saved: {len(processed_cases)} cases processed")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def save_single_case_to_json(case_idx, processed_case):
    """Save a single completed case immediately to the ORIGINAL JSON file."""
    global output_file, data_to_process
    
    try:
        # Update the data in memory
        data_to_process[case_idx] = processed_case
        
        # Atomic write: write to temporary file first, then rename OVER ORIGINAL
        temp_file = output_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_process, f, indent=2, ensure_ascii=False)
        
        # Atomic rename (this is atomic on most filesystems) - OVERWRITES ORIGINAL
        os.replace(temp_file, output_file)
        
        logger.info(f"💾 Case {case_idx} immediately saved to ORIGINAL file: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save case {case_idx} to ORIGINAL JSON: {e}")
        # Try to clean up temp file if it exists
        temp_file = output_file + '.tmp'
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def save_output_files():
    """Save output files - OVERWRITES ORIGINAL FILE."""
    global processed_cases, failed_cases, data_to_process, output_file
    
    if not data_to_process:
        return
        
    try:
        # Create processed data
        processed_data = []
        for i, case in enumerate(data_to_process):
            if i in processed_cases:
                processed_data.append(processed_cases[i])
            else:
                processed_data.append(case)  # Keep original if not processed
        
        # Save processed data - OVERWRITES ORIGINAL FILE
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Generate and save report
        total_time = time.time() - start_time if start_time else 0
        success_rate = (len(processed_cases) / len(data_to_process)) * 100 if data_to_process else 0
        
        report = {
            'summary': {
                'total_cases': len(data_to_process),
                'successful_cases': len(processed_cases),
                'failed_cases': len(failed_cases),
                'success_rate': f"{success_rate:.2f}%",
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'processing_rate_cases_per_second': len(processed_cases) / total_time if total_time > 0 else 0,
                'worker_count': MAX_WORKERS,
                'cases_per_worker': CASES_PER_WORKER,
                'completion_status': 'completed' if len(processed_cases) == len(data_to_process) else 'partial'
            },
            'failed_cases': failed_cases,
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = output_file.replace('.json', '_max_concurrency_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Output files saved:")
        logger.info(f"   - ORIGINAL FILE OVERWRITTEN: {output_file}")
        logger.info(f"   - Report: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to save output files: {e}")

def load_json(file_path):
    """Load JSON file - optimized for speed and robust parsing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try to parse as single JSON object first
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "messages" in data:
                # Single JSON object with messages
                return [data]
            elif isinstance(data, list):
                # JSON array
                return data
            else:
                # Single JSON object without messages
                return [data]
        except json.JSONDecodeError:
            # Try to parse as JSONL (one JSON object per line)
            lines = content.split('\n')
            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    try:
                        parsed = json.loads(line)
                        data.append(parsed)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {i+1}: {e}")
                        logger.warning(f"Line content: {line[:100]}...")
                        continue
            return data
            
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def process_single_case_robust(case_data):
    """Process a single case with BULLETPROOF error handling and retries."""
    case, case_idx, worker_id = case_data
    
    try:
        # Initialize pipeline for this worker
        config = PipelineConfig(
            max_features=6,
            confidence_threshold=0.5,
            temperature=0.8,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
            asr=False
        )
        pipeline = GranularSpeechPipeline(config)
        clarification_extractor = ClarificationExtractor()
        
        # Extract tools
        function_docs = case.get("tools", [])
        if not isinstance(function_docs, list):
            function_docs = []
        
        # Process each message in the case
        for msg in case.get("messages", []):
            if msg.get("role") == "user" and "content" in msg:
                try:
                    # Process transcript with retries
                    transcript = None
                    for attempt in range(MAX_RETRIES):
                        try:
                            transcript = pipeline.transform_text(msg["content"])
                            break
                        except Exception as e:
                            if attempt == MAX_RETRIES - 1:
                                logger.warning(f"Worker {worker_id}: Failed to process transcript after {MAX_RETRIES} attempts: {e}")
                                transcript = msg["content"]  # Fallback to original
                            else:
                                logger.info(f"Worker {worker_id}: Retrying transcript processing (attempt {attempt + 1})")
                                time.sleep(RETRY_DELAY)
                    
                    # Extract transcript string
                    if isinstance(transcript, dict) and "final" in transcript:
                        transcript_str = transcript["final"]
                    elif isinstance(transcript, str):
                        transcript_str = transcript
                    elif isinstance(transcript, list) and transcript and isinstance(transcript[0], dict) and "final" in transcript[0]:
                        transcript_str = transcript[0]["final"]
                    else:
                        transcript_str = str(transcript)
                    
                    # Generate clarifications with retries
                    clarifications = []
                    for attempt in range(MAX_RETRIES):
                        try:
                            clarifications = clarification_extractor.generate_clarifications(transcript_str, function_docs)
                            break
                        except Exception as e:
                            if attempt == MAX_RETRIES - 1:
                                logger.warning(f"Worker {worker_id}: Failed to generate clarifications after {MAX_RETRIES} attempts: {e}")
                                clarifications = []
                            else:
                                logger.info(f"Worker {worker_id}: Retrying clarification generation (attempt {attempt + 1})")
                                time.sleep(RETRY_DELAY)
                    
                    # Update message
                    if "eot" in msg:
                        new_msg = {}
                        for k, v in msg.items():
                            if k == "eot":
                                new_msg["transcript"] = transcript_str
                                new_msg["clarifications"] = clarifications
                            new_msg[k] = v
                        msg.clear()
                        msg.update(new_msg)
                    else:
                        msg["transcript"] = transcript_str
                        msg["clarifications"] = clarifications
                        
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Failed to process message in case {case_idx}: {e}")
                    # Keep original message on error
                    continue
        
        logger.info(f"Worker {worker_id}: Completed case {case_idx}")
        return case_idx, case, worker_id, "success"
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Critical failure on case {case_idx}: {e}")
        return case_idx, case, worker_id, "failed"

def check_json_file_status():
    """Check the current status of the ORIGINAL JSON file."""
    global output_file, data_to_process
    
    if not os.path.exists(output_file):
        logger.info(f"📁 Original file not yet created: {output_file}")
        return
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # Count processed cases (those with transcript/clarifications)
        processed_count = 0
        for case in current_data:
            if case.get("messages"):
                for msg in case["messages"]:
                    if msg.get("role") == "user" and "transcript" in msg and "clarifications" in msg:
                        processed_count += 1
                        break
        
        total_cases = len(current_data)
        logger.info(f"📊 ORIGINAL FILE STATUS:")
        logger.info(f"   📁 File: {output_file}")
        logger.info(f"   📊 Total cases: {total_cases}")
        logger.info(f"   ✅ Processed cases: {processed_count}")
        logger.info(f"   📈 Progress: {processed_count}/{total_cases} ({processed_count/total_cases*100:.1f}%)")
        logger.info(f"   ⚠️  This file will be OVERWRITTEN with enhanced data")
        
    except Exception as e:
        logger.error(f"Failed to check ORIGINAL JSON file status: {e}")

def main(file_path, test_mode=False):
    global processed_cases, failed_cases, start_time, checkpoint_file, output_file, data_to_process
    
    # Set up signal handlers and exit handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(save_checkpoint)
    atexit.register(save_output_files)
    
    start_time = time.time()
    
    # Set up file paths - EDIT ORIGINAL FILE DIRECTLY
    checkpoint_file = file_path.replace('.json', '_checkpoint.pkl')
    output_file = file_path  # Use original file path - will overwrite original
    
    # Load data
    logger.info(f"Loading data from {file_path}")
    data = load_json(file_path)
    logger.info(f"Loaded {len(data)} cases")
    
    # Try to load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            processed_cases = checkpoint.get('processed_cases', {})
            failed_cases = checkpoint.get('failed_cases', [])
            logger.info(f"✅ Loaded checkpoint: {len(processed_cases)} cases already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            processed_cases = {}
            failed_cases = []
    
    # Determine processing parameters
    if test_mode:
        data_to_process = data[:100]  # Test with 100 cases
        worker_count = 100  # 100 workers for test
    else:
        data_to_process = data
        worker_count = min(MAX_WORKERS, len(data_to_process))
    
    logger.info(f"Processing {len(data_to_process)} cases with {worker_count} workers")
    logger.info(f"Each worker will process {CASES_PER_WORKER} case(s) simultaneously")
    logger.info(f"Checkpointing every {CHECKPOINT_INTERVAL} cases")
    logger.info(f"Force saving every {FORCE_SAVE_INTERVAL} seconds")
    logger.info(f"⚠️  WARNING: Will OVERWRITE original file: {file_path}")
    
    # Prepare case assignments - each worker gets exactly one case
    case_assignments = []
    for i, case in enumerate(data_to_process):
        if i not in processed_cases:  # Skip already processed cases
            worker_id = i % worker_count
            case_assignments.append((case, i, worker_id))
    
    logger.info(f"Processing {len(case_assignments)} remaining cases")
    
    # Process all cases with maximum concurrency
    completed_count = len(processed_cases)
    failed_count = len(failed_cases)
    last_checkpoint_time = time.time()
    last_force_save_time = time.time()
    
    logger.info(f"Starting parallel processing with {worker_count} workers...")
    
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        # Submit all cases simultaneously
        future_to_case = {
            executor.submit(process_single_case_robust, case_data): case_data 
            for case_data in case_assignments
        }
        
        # Process completed futures as they finish
        for future in as_completed(future_to_case):
            case_idx, processed_case, worker_id, status = future.result()
            
            if status == "success":
                processed_cases[case_idx] = processed_case
                completed_count += 1
                logger.info(f"✅ Case {case_idx} completed by worker {worker_id} ({completed_count}/{len(data_to_process)})")
                save_single_case_to_json(case_idx, processed_case) # Save immediately
                
                # Show real-time progress
                if completed_count % 10 == 0:  # Every 10 cases
                    logger.info(f"📊 REAL-TIME PROGRESS: {completed_count}/{len(data_to_process)} cases saved to ORIGINAL file")
            else:
                failed_cases.append({
                    'case_idx': case_idx,
                    'worker_id': worker_id,
                    'error': 'Processing failed',
                    'timestamp': datetime.now().isoformat()
                })
                failed_count += 1
                logger.warning(f"❌ Case {case_idx} failed in worker {worker_id} ({failed_count} failures)")
            
            # Checkpoint periodically
            current_time = time.time()
            if completed_count % CHECKPOINT_INTERVAL == 0:
                save_checkpoint()
                last_checkpoint_time = current_time
            
            # Force save periodically (every 5 minutes)
            if current_time - last_force_save_time >= FORCE_SAVE_INTERVAL:
                save_output_files()
                last_force_save_time = current_time
                logger.info(f"🔄 Force save completed at {completed_count}/{len(data_to_process)} cases")
                
                # Check JSON file status
                check_json_file_status()
            
            # Progress update
            if (completed_count + failed_count) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (completed_count + failed_count) / elapsed if elapsed > 0 else 0
                eta = (len(data_to_process) - completed_count - failed_count) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed_count + failed_count}/{len(data_to_process)} cases processed. "
                          f"Rate: {rate:.2f} cases/sec. ETA: {eta/60:.1f} minutes")
    
    # Final processing complete
    total_time = time.time() - start_time
    success_rate = (completed_count / len(data_to_process)) * 100 if data_to_process else 0
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"Total cases: {len(data_to_process)}")
    logger.info(f"Successful: {completed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Total time: {total_time/60:.2f} minutes")
    logger.info(f"Average rate: {len(data_to_process)/total_time:.2f} cases/second")
    logger.info("=" * 60)
    
    # Final save - OVERWRITES ORIGINAL FILE
    save_output_files()
    
    # Clean up checkpoint file on success
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("Checkpoint file cleaned up")
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"⚠️  ORIGINAL FILE OVERWRITTEN: {file_path} now contains enhanced data with transcripts & clarifications")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BULLETPROOF Maximum Concurrency API Generation Pipeline')
    parser.add_argument('--file_path', type=str, required=True, help='Path to api_gen JSON to process')
    parser.add_argument('--test', action='store_true', help='Run test mode: 100 workers, 100 cases')
    
    args = parser.parse_args()
    
    try:
        main(args.file_path, test_mode=args.test)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        save_checkpoint()
        save_output_files()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        save_checkpoint()
        save_output_files()
        sys.exit(1)
