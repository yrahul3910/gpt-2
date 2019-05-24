import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Filter required messages from Telegram dump',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--input', metavar='INPUT', type=str, required=True,
                    help='Input JSON file')
parser.add_argument('--output', metavar='OUTPUT', type=str, required=True,
                    help='Output file name')
parser.add_argument('--name', metavar='NAME', type=str, required=True,
                    help='Contact Telegram name (not username)')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        text = f.read()

    # Get the list of all chats as JSON
    data = json.loads(text)
    chats = data['chats']['list']

    # Check names against 'name' key in the list
    def check_name(obj, name):
        if 'name' in obj.keys():
            return obj['name'] == name
        return False

    name_matches = list(map(lambda x: check_name(x, args.name), chats))

    try:
        index = name_matches.index(True)
        required_chat = chats[index]
        messages = required_chat['messages']
        messages = list(filter(lambda x: x['type'] == 'message', messages))

        # Format a message inside the list
        def format_message(obj):
            result = obj['from'] + ': '

            # Happens when there are hashtags inside a text
            if type(obj['text']) == list:
                for el in obj['text']:
                    if type(el) == dict:
                        # Parse hashtags
                        result += el['text']
                    else:
                        # Parse text
                        result += el
            else:
                result += obj['text']

            return result

        formatted_messages = list(map(lambda x: format_message(x), messages))

        with open(args.output, 'w') as f:
            f.write('\n'.join(formatted_messages))
    except ValueError:
        print('No chats for name', args.name, 'exist in the dump.')
