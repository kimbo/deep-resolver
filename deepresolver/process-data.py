import json
import sys
import argparse
import dns.message
import base64

import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-o', '--output', default='data.jsonl')
    args = parser.parse_args()
    with open(args.filename, 'r') as inp, open(args.output, 'w') as out:
        for line in tqdm.tqdm(inp, total=828):
            exec(line)
            things = {k: v for k, v in locals().items() if 'alexa_data' in k}
            for k, v in things.items():
                for item in v:
                    s = json.dumps([base64.b64encode(item['query']).decode(), base64.b64encode(item['response']).decode()])
                    print(s, file=out)
            for k in things.keys():
                del locals()[k]

if __name__ == '__main__':
    main()
