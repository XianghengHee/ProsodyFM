# %%
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
from tqdm.auto import tqdm

# %%
train_file_list = './filelists/libritts_audio_sid_text_train_filelist.txt'
val_file_list = './filelists/libritts_audio_sid_text_val_filelist.txt'
test_file_list = './filelists/libritts_audio_sid_text_test_filelist.txt'

# %%
def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

# %%
processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")
model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en").to("cuda:0")

# %%
def generate_transcription(audio, gpu=False):
    """Generate a transcription from audio using a pre-trained model

    Args:
    audio: The audio to be transcribed
    gpu: Whether to use GPU or not. Defaults to False.

    Returns:
    transcription: The transcribed text
    """
    # Preprocess audio and return tensors
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

    # Assign inputs to GPU or CPU based on argument
    if gpu:
        input_features = inputs.input_features.cuda()
    else:
        input_features = inputs.input_features

    # Generate transcribed ids
    generated_ids = model.generate(inputs=input_features, max_length=250)

    # Decode generated ids and replace special tokens
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True, output_word_offsets=True)[0].replace('!!!!!', '@')

    return transcription

# %%
filepaths_and_text = parse_filelist(val_file_list)

output_filepaths_and_text = []
for info_list in tqdm(filepaths_and_text):
    output_info_list = []
    
    audio_path = info_list[0]
    spk_id = info_list[1]
    text = info_list[2]
    
    y, sr = librosa.load(audio_path)
    audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    transcript = generate_transcription(audio, gpu=True)
    
    output_info_list.append(audio_path)
    output_info_list.append(spk_id)
    output_info_list.append(transcript)

    output_filepaths_and_text.append(output_info_list)

with open('./filelists/libritts_audio_sid_text_val_filelist_with_boundary.txt', 'w') as file:
    for item in output_filepaths_and_text:
        # Join the elements of the sublist with '|' and write to the file
        file.write('|'.join(item) + '\n')

filepaths_and_text = parse_filelist(test_file_list)

output_filepaths_and_text = []
for info_list in tqdm(filepaths_and_text):
    output_info_list = []
    
    audio_path = info_list[0]
    spk_id = info_list[1]
    text = info_list[2]
    
    y, sr = librosa.load(audio_path)
    audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    transcript = generate_transcription(audio, gpu=True)
    
    output_info_list.append(audio_path)
    output_info_list.append(spk_id)
    output_info_list.append(transcript)

    output_filepaths_and_text.append(output_info_list)

with open('./filelists/libritts_audio_sid_text_test_filelist_with_boundary.txt', 'w') as file:
    for item in output_filepaths_and_text:
        # Join the elements of the sublist with '|' and write to the file
        file.write('|'.join(item) + '\n')

filepaths_and_text = parse_filelist(train_file_list)

output_filepaths_and_text = []
for info_list in tqdm(filepaths_and_text):
    output_info_list = []
    
    audio_path = info_list[0]
    spk_id = info_list[1]
    text = info_list[2]
    
    y, sr = librosa.load(audio_path)
    audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    transcript = generate_transcription(audio, gpu=True)
    
    output_info_list.append(audio_path)
    output_info_list.append(spk_id)
    output_info_list.append(transcript)

    output_filepaths_and_text.append(output_info_list)

# %%
with open('./filelists/libritts_audio_sid_text_train_filelist_with_boundary.txt', 'w') as file:
    for item in output_filepaths_and_text:
        # Join the elements of the sublist with '|' and write to the file
        file.write('|'.join(item) + '\n')
