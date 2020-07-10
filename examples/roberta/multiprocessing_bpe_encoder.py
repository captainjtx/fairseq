#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

import time, math

def get_subsample_filename(file_name, subsample_ratio):
    if subsample_ratio <= 0:
        return file_name
    else:
        return file_name + ".subsample_ratio_{}".format(subsample_ratio)

def subsample(input_file, subsample_ratio):
    """
    We subsample the file directly and then save to a file named
    as <original_name>.sub_sample_ratio_<subsample ratio>
    Note each paragraph starts with
    empty line
    title line
    empty line
    content lines
    We downsample at paragraph boundary
    """
    assert subsample_ratio > 0.0, "subsample ratio needs to be > 0."
    assert subsample_ratio <= 1.0
    sample_every_n = math.floor(1.0 / subsample_ratio)
    empty_cnt = 0
    do_sample = True
    output_file = get_subsample_filename(input_file, subsample_ratio)
    with open(input_file, "r") as r:
        with open(output_file, "w") as w:
            for line in r.readlines():
                # note in the input file, the empty line has length 2
                if len(line) == 2:
                    empty_cnt += 1
                if empty_cnt % 2 == 1:
                    if ((empty_cnt - 1) / 2) % sample_every_n == 0:
                        do_sample = True
                    else:
                        do_sample = False
                if do_sample:
                    w.write(line)


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    # add the subsampling strategy
    # note here we make the default value 1.0 to make sure
    # the scripts executes with their old naming convention.
    # When the ratio is explicitly given, it needs to be > 0.
    # Then it will generate output with the our own naming convention.
    parser.add_argument("--subsample-ratio", type=float, default=0.0,
        help="subsampling ratio: 1.0 for sample the full set.")


    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    # subsample and write to files if needed
    if args.subsample_ratio > 0:
        for input in args.inputs:
            subsample(input, args.subsample_ratio)

    if args.workers == 1:
        # Support single process for the ease of debugging
        bpe = get_encoder(args.encoder_json, args.vocab_bpe)
        for input, output in zip(args.inputs, args.outputs):
            istream = sys.stdin
            ostream = sys.stdout
            if input != '-':
                istream = open(get_subsample_filename(input, args.subsample_ratio), 'r', encoding="utf-8")
            if output != '-':
                ostream = open(get_subsample_filename(output, args.subsample_ratio), 'w', encoding="utf-8")
            enc_lines = []
            for line in istream:
                line = line.strip()
                if len(line) == 0 and not args.keep_empty:
                    enc_lines = None
                    break
                ids = bpe.encode(line)
                tokens = list(map(str, ids))
                enc_lines.append(" ".join(tokens))
            if enc_lines is not None:
                for enc_line, output_h in zip(enc_lines, ostream):
                    print(enc_line, file=output_h)

    else:
        with contextlib.ExitStack() as stack:
            inputs = [
                stack.enter_context(open(get_subsample_filename(input, 
                    args.subsample_ratio), "r", encoding="utf-8"))
                if input != "-" else sys.stdin
                for input in args.inputs
            ]
            outputs = [
                stack.enter_context(open(get_subsample_filename(output,
                    args.subsample_ratio), "w", encoding="utf-8"))
                if output != "-" else sys.stdout
                for output in args.outputs
            ]

            encoder = MultiprocessingEncoder(args)
            pool = Pool(args.workers, initializer=encoder.initializer)
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

            stats = Counter()
            for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                if filt == "PASS":
                    for enc_line, output_h in zip(enc_lines, outputs):
                        print(enc_line, file=output_h)
                else:
                    stats["num_filtered_" + filt] += 1
                if i % 10000 == 0:
                    print("processed {} lines".format(i), file=sys.stderr)

            for k, v in stats.most_common():
                print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
