import json

def transform_and_save_json(input_file, output_file):
    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data_array = json.load(f)
    
    result = []
    
    # Process each object in the array
    for data in data_array:
        timestamp = data['timestamp']
        
        # Process each key-value pair except timestamp
        for key, value in data.items():
            if key != 'timestamp':
                # Split the key by underscore
                parts = key.split('_')
                if len(parts) >= 3:
                    region = parts[0]
                    sensor_type = parts[1]
                    measurement = '_'.join(parts[2:])
                    
                    # Create new entry
                    new_entry = {
                        'timestamp': timestamp,
                        'region': region,
                        'sensor_type': sensor_type,
                        'measurement': measurement,
                        'value': value
                    }
                    result.append(new_entry)
    
    # Save transformed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Transformed data has been saved to {output_file}")

# Example usage
input_file = r'D:\Backend\ISSC-SWS-Backend\app\data\algorithm11\TimeMixer\predictions_TimeMixer_auto.json'  # 替换为你的输入文件路径
output_file = r'D:\Backend\ISSC-SWS-Backend\app\data\algorithm11\TimeMixer\predictions_TimeMixer_auto1.json'  # 替换为你想要保存的输出文件路径

transform_and_save_json(input_file, output_file)