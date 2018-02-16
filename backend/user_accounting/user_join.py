import json

def save_user_join(data_path, output_path):
    output = "user,timestamp\n"
    with open(data_path) as data_file:
        data = json.load(data_file)
        for message in data['messages']:
            if 'subtype' in message:
                if message['subtype'] == 'channel_join':
                    output += message['user']
                    output += ","
                    output += message['ts']
                    output += "\n"
    out_file = open(output_path, 'w+')
    out_file.write(output)
    out_file.close()


if __name__ == "__main__":
    save_user_join('general.json', '../export/user_join.csv')
