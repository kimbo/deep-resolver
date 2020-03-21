import base64
import json
from collections import defaultdict

import tqdm
import dns.message

if __name__ == '__main__':
    rcode_dict = defaultdict(int)
    with open('data.jsonl', 'r') as fp:
        for line in tqdm.tqdm(fp, total=4_090_421):
            query, response = json.loads(line)
            # query = dns.message.from_wire(query)
            response = dns.message.from_wire(base64.b64decode(response.encode()))
            rcode_dict[dns.rcode.to_text(response.rcode())] += 1
    with open('rcode-stats.json', 'w') as fp:
        json.dump(rcode_dict, fp)

