import socket
import sys
import argparse
import logging

import dns.message
import dns.rdatatype

import torch

from train import bytes_to_tensor, DNSModel, tensor_to_bytes

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

class DeepResolver:
    def __init__(self, fname, port=5353, bind_addr=None):
        self.model = torch.load(fname)
        self.model.eval()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if not bind_addr:
            bind_addr = '0.0.0.0'
        self._sock.bind((bind_addr, port))
        logger.info('Deep resolver listening at {}#{}'.format(bind_addr, port))

    def handle_queries(self):
        while True:
            wire, (addr, port) = self._sock.recvfrom(65536)
            tensor = bytes_to_tensor(wire, self.model.linear.in_features)
            while True:
                try:
                    out = self.model(tensor.float())
                    wire = tensor_to_bytes(out)
                    answer = dns.message.from_wire(wire, ignore_trailing=True)
                    self._sock.sendto(answer.to_wire(), (addr, port))
                    break
                except Exception as e:
                    logger.exception(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_fname')
    parser.add_argument('-p', '--port', type=int)
    parser.add_argument('-b', '--bind')
    args = parser.parse_args()

    deep_resolver = DeepResolver(args.model_fname, args.port, args.bind)
    deep_resolver.handle_queries()

if __name__ == '__main__':
    main()