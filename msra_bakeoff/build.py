import xml.etree.ElementTree as ET
import os
import argparse
import yaml
from collections import defaultdict
import sys
import re


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


def word_category_tag_stream(sentence_node):
    for word_node in sentence_node:
        word = word_node.text
        if word is None:
            for entity_node in word_node:
                word, category, tag = entity_node.text, entity_node.tag, entity_node.attrib["TYPE"]
                yield word, category, tag
        else:
            yield word, None, None


def converted_word_tag_stream(word_category_tag_stream, tag_mapping):
    """
    Leave word as is, convert tag according to a mapping specified.
    e.g.
    <w>word<w> => (word, tag_mapping[None][None])
    <w><NAMEX TYPE="LOCATION">word</NAMEX></word> => (word, tag_mapping["NAMEX"]["LOCATION"]
    :param word_category_tag_stream: (word, category, tag) stream.
    :param tag_mapping: dict. Converted tag would be tag_mapping[category][tag]
    :return:
    """
    for word, category, tag in word_category_tag_stream:
        yield word, tag_mapping[category][tag]



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


def char_bmeso_tag_stream(word_tag_pairs, non_entity_tag="O"):
    """
    Yield char-BMESO_tag pairs.
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
                if i == 0:
                    bmeso_tag = ("B-%s" if len(word) > 1 else "S-%s") % tag
                elif i == len(word) - 1:
                    bmeso_tag = "E-%s" % tag
                else:
                    bmeso_tag = "M-%s" % tag
                yield ch, bmeso_tag


class SpokenLanguageWordProcessor:
    NUMBER_MAPPING = {"0": "零", "０": "零", "〇": "零", "○": "零",
                      "1": "一", "１": "一",
                      "2": "二", "２": "二",
                      "3": "三", "３": "三",
                      "4": "四", "４": "四",
                      "5": "五", "５": "五",
                      "6": "六", "６": "六",
                      "7": "七", "７": "七",
                      "8": "八", "８": "八",
                      "9": "九", "９": "九"}
    NUMBERS = "零一二三四五六七八九"

    def __init__(self):
        pass

    def process(self, word_category_tag_stream):
        for word, category, tag in word_category_tag_stream:
            if tag == "PHONE":
                word = word.replace("一", "幺")
            if tag == "DATE" and re.search("[零一二三四五六七八九]日", word):
                print("Word before conversion:", word)
                word = word[:-1] + "号"
                print("Word after conversion:", word)
            if tag == "TIM":
                match = re.search("[零一二三四五六七八九]时", word)
                if match:
                    replace_idx = match.span(0)[-1]
                print("Word before conversion:", word)
                word = word[:replace_idx] + "点" + word[:replace_idx + 1]
                print("Word after conversion:", word)
            if tag == "MON":
                match = re.search("[零一二三四五六七八九]元", word)
                if match:
                    replace_idx = match.span(0)[-1]
                print("Word before conversion:", word)
                word = word[:replace_idx] + "块" + word[:replace_idx + 1]
                print("Word after conversion:", word)
            word = self._safe_replace(word, self.__class__.NUMBER_MAPPING)
            if len(word) > 0:
                yield word, category, tag

    @classmethod
    def _safe_replace(cls, word, mapping):
        return "".join(map(lambda x: mapping[x] if x in mapping else x, word))


class DefaultProcessor:
    """Do nothing, leave everything as is."""

    def __init__(self):
        pass

    def process(self, word_category_tag_stream):
        for item in word_category_tag_stream:
            yield item


def main(args):
    with open(os.path.normpath(args.config_file), "r") as f:
        config = yaml.load(f)
    tag_mapping = get_tag_mapping(config["tag-params"])
    i_file_paths = config["file-params"]["input-file"]
    o_file_paths = config["file-params"]["output-file"]
    if "processor" in config:
        processor = {"speech2text": SpokenLanguageWordProcessor}[config["processor"]]()
    else:
        processor = DefaultProcessor()
    ch_tag_stream = {"bio": char_bio_tag_stream, "bmeso": char_bmeso_tag_stream}[config["tag-params"]["tag-format"].lower()]
    for i_file_path, o_file_path in zip(i_file_paths, o_file_paths):
        sys.stdout.write("Processing file:\n{}\n".format(i_file_path))
        with open(os.path.normpath(i_file_path), "r", encoding="gb18030") as f:
            text = f.read()
        root = ET.fromstring(text)
        with open(os.path.normpath(o_file_path), "w", encoding="utf-8") as f:
            for sentence_node in root:
                # print("\n".join(map(lambda *x: "\t".join(*x), word_tag_stream(sentence_node, tag_mapping))))
                sentence_dump = \
                    "\n".join(["{}\t{}".format(ch, tag) for ch, tag in ch_tag_stream(converted_word_tag_stream(processor.process(word_category_tag_stream(sentence_node)), tag_mapping))])
                f.write(sentence_dump)
                f.write("\n\n")
        sys.stdout.write("Saving parsed file to:\n{}\n".format(o_file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
