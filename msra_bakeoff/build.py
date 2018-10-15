import xml.etree.ElementTree as ET
import os
import argparse
import yaml
from collections import defaultdict
import sys


def get_tag_mapping(tag_params):
    def get_lambda(value):
        """
        Directly use defaultdict(lambda : value) will make the expressions subject to change
        in a for loop, That's why a function is defined here
        :param value:
        :return:
        """
        return lambda: value
    tag_mapping = {None: defaultdict(get_lambda(tag_params["O"]))}
    for entity_class in ("NAMEX", "TIMEX", "NUMEX", "MEASUREX", "ADDREX"):
        if entity_class in tag_params:
            if isinstance(tag_params[entity_class], str):
                tag_mapping[entity_class] = defaultdict(get_lambda(tag_params[entity_class]))
            elif isinstance(tag_params[entity_class], dict):
                tag_mapping[entity_class] = defaultdict(get_lambda(tag_params["O"]))
                for sub_entity_class in tag_params[entity_class]:
                    tag_mapping[entity_class][sub_entity_class] = tag_params[entity_class][sub_entity_class]
            else:
                raise ValueError("Unsupported tag mapping for {}".format(tag_params[entity_class]))
        else:
            tag_mapping[entity_class] = defaultdict(get_lambda(tag_params["O"]))
    del entity_class
    # parse the special vt token that as <TIMEX/>
    # e.g.
    # <sentence>
    #   <w><TIMEX >date1</TIMEX><w/>
    #   <w>to</w>
    #   <W><vt>DATE2<vt></w>
    # <sentence/>
    tag_mapping["vt"] = tag_mapping["TIMEX"]
    return tag_mapping


def word_tag_stream(sentence_node, tag_mapping):
    for word_node in sentence_node:
        word = word_node.text
        if word is None:
            for entity_node in word_node:
                word, tag = entity_node.text, tag_mapping[entity_node.tag][entity_node.attrib["TYPE"]]
                yield word, tag
        else:
            tag = tag_mapping[None][None]
            yield word, tag


def char_bio_tag_stream(word_tag_pairs, non_entity_tag="O"):
    """
    Yield char-bio-tag pairs.
    :param word_tag_pairs: Iterable word-tag sequence. e.g. [(word1, O), (word2, ORG), ...]
    :param non_entity_tag: Non-entity tag.
    :return:
    """
    for word, tag in word_tag_pairs:
        if tag == non_entity_tag:
            for ch in word:
                yield ch, non_entity_tag
        else:
            for i, ch in enumerate(word):
                bio_tag = "B-%s" if i == 0 else "I-%s"
                bio_tag = bio_tag % tag
                yield ch, bio_tag


def main(args):
    with open(os.path.normpath(args.config_file), "r") as f:
        config = yaml.load(f)
    tag_mapping = get_tag_mapping(config["tag-params"])
    i_file_paths = config["file-params"]["input-file"]
    o_file_paths = config["file-params"]["output-file"]
    for i_file_path, o_file_path in zip(i_file_paths, o_file_paths):
        sys.stdout.write("Processing file:\n{}\n".format(i_file_path))
        with open(os.path.normpath(i_file_path), "r", encoding="gb18030") as f:
            text = f.read()
        root = ET.fromstring(text)
        with open(os.path.normpath(o_file_path), "w", encoding="utf-8") as f:
            for sentence_node in root:
                # print("\n".join(map(lambda *x: "\t".join(*x), word_tag_stream(sentence_node, tag_mapping))))
                sentence_dump = \
                    "\n".join(["{}\t{}".format(ch, bio_tag) for ch, bio_tag in char_bio_tag_stream(word_tag_stream(sentence_node, tag_mapping))])
                f.write(sentence_dump)
                f.write("\n\n")
        sys.stdout.write("Saving parsed file to:\n{}\n".format(o_file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
