import argparse
import base64
import json
import sys
import random
import logging
import threading

import dns.query
import dns.message
import dns.rdatatype

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

rdtypes = [dns.rdatatype.A, dns.rdatatype.AAAA, dns.rdatatype.MX, dns.rdatatype.TXT, dns.rdatatype.SRV]
resolvers = ['1.1.1.1', '8.8.8.8', '9.9.9.9', '208.67.222.222', '64.6.64.6']

lock = threading.Lock()

def query_worker(infile: str, outfile: str):
    def read_data():
        _results = []
        with open(infile, 'r') as fp:
            for line in fp:
                qname = line.strip().split(',')[1] + '.'
                for t in rdtypes:
                    q = dns.message.make_query(qname, t)
                    q.use_edns(True)
                    where = random.choice(resolvers)
                    try:
                        ans = dns.query.udp(q, where=where, timeout=2)
                    except dns.exception.Timeout:
                        continue
                    except Exception as e:
                        logger.exception(e)
                        continue
                    else:
                        _results.append((base64.b64encode(q.to_wire()).decode(), base64.b64encode(ans.to_wire()).decode()))
                        logger.info('{} -> {} ? {}'.format(qname, where, dns.rcode.to_text(ans.rcode())))
        return _results
    try:
        results = read_data()
    except Exception as e:
        logger.exception(e)
    else:
        with lock, open(outfile, 'a') as out:
            for r in results:
                print(json.dumps(r), file=out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain_files', nargs='+', help='Files with domains in them to resolve')
    parser.add_argument('-o', '--output', required=True, help='Output file to store queries and responses in')
    args = parser.parse_args()
    threads = []
    for file in args.domain_files:
        t = threading.Thread(target=query_worker, args=(file, args.output))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()