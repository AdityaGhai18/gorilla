import os
import json
import glob

def process_folders():
    # Get all folders in current directory
    folders = [f for f in os.listdir('.') if os.path.isdir(f)]
    all_entries = []
    func = None
    for folder in folders:
        input_file = os.path.join(folder, 'input.jsonl')
        
        # Check if input.jsonl exists in this folder
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    for count, line in enumerate(f):
                        try:
                            entry = json.loads(line.strip())
                            # Assign folder name as ID
                            print([item["name"] for item in entry["function"]])
            #                 if entry["function"] != func:
            #                     print("1")
            #                     func = entry["function"]
            #                 all_entries.append({
            #                     "id": f"live_multiple_{folder}-{count}",
            #                     "question": 
            #                     [[{"role": "user", "content": entry["question"]}]],
            #                     "function": entry["function"],
            #                     "massive_id": entry["id"],
            #                 })
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line in {input_file}")
                            continue
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
    
    # Write all entries to a single output file
    # output_file = 'combined_output.jsonl'
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for entry in all_entries:
    #         f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # print(f"Successfully combined {len(all_entries)} entries from {len(folders)} folders into {output_file}")

# def process_folders():
#     # Get all folders in current directory
#     folders = [f for f in os.listdir('.') if os.path.isdir(f)]
#     all_entries = []
    
#     for folder in folders:
#         input_file = os.path.join(folder, 'input.jsonl')
        
#         # Check if input.jsonl exists in this folder
#         if os.path.exists(input_file):
#             try:
#                 with open(input_file, 'r', encoding='utf-8') as f:
#                     for count, line in enumerate(f):
#                         try:
#                             entry = json.loads(line.strip())
#                             # Assign folder name as ID
#                             print(entry["language"])
#                             all_entries.append({
#                                 "id": f"live_multiple_{folder}-{count}",
#                                 "ground_truth": [entry["gt"]]
#                             })
#                         except json.JSONDecodeError:
#                             print(f"Warning: Skipping invalid JSON line in {input_file}")
#                             continue
#             except Exception as e:
#                 print(f"Error processing {input_file}: {e}")
    
#     # Write all entries to a single output file
#     output_file = 'gt.jsonl'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for entry in all_entries:
#             f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
#     print(f"Successfully combined {len(all_entries)} entries from {len(folders)} folders into {output_file}")

if __name__ == "__main__":
    print(len(['alarm.query', 'alarm.remove', 'alarm.set', 'audio.volume_down', 'audio.volume_mute', 'audio.volume_up', 'calendar.query', 'calendar.remove', 'calendar.set', 'cooking.recipe', 'datetime.convert', 'datetime.query', 'email.addcontact', 'email.query', 'email.querycontact', 'email.sendemail', 'general.joke', 'iot.cleaning', 'iot.coffee', 'iot.hue_lightchange', 'iot.hue_lightdim', 'iot.hue_lightoff', 'iot.hue_lighton', 'iot.hue_lightup', 'iot.wemo_off', 'iot.wemo_on', 'lists.createoradd', 'lists.query', 'lists.remove', 'music.dislikeness', 'music.likeness', 'music.query', 'news.query', 'play.audiobook', 'play.game', 'play.music', 'play.podcasts', 'play.radio', 'qa.currency', 'qa.definition', 'qa.factoid', 'qa.maths', 'qa.stock', 'recommendation.events', 'recommendation.locations', 'recommendation.movies', 'social.post', 'social.query', 'takeaway.order', 'takeaway.query', 'transport.query', 'transport.taxi', 'transport.ticket', 'transport.traffic', 'weather.query']))