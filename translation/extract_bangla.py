
import json

def extract_bangla(input_file, output_raw_file, output_polished_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_raw_file, 'w', encoding='utf-8') as f_raw, \
         open(output_polished_file, 'w', encoding='utf-8') as f_polished:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            row_id = data.get('row_id')
            
            # Raw Bangla
            if 'bangla_raw' in data:
                raw_data = {
                    'row_id': row_id,
                    'bangla': data['bangla_raw']
                }
                f_raw.write(json.dumps(raw_data, ensure_ascii=False) + '\n')
            
            # Polished Bangla
            if 'bangla_polished' in data:
                polished_data = {
                    'row_id': row_id,
                    'bangla': data['bangla_polished']
                }
                f_polished.write(json.dumps(polished_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # For test files
    extract_bangla(
        'polished_test.jsonl',
        'bangla_raw_test.jsonl',
        'bangla_polished_test.jsonl'
    )
    print("Extraction complete!")
