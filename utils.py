import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq, suffixes=("PER", "LOC", "ORG"), strict=True):
    """
    :param tag_seq: A list containing B/I-Entity and other tags.
    :param char_seq: Character sequence of the same length as tag_seq.
    :param suffixes: A iterable yielding type of entities to be detected. Default
    :param strict: bool. If true, raise ValueError when tag_seq is invalid.
    :return: List with the same length of number of entities to be detected.
    """
    return tuple(get_BIO_entity(tag_seq, char_seq, suffix, strict) for suffix in suffixes)


def get_BIO_entity(tag_seq, char_seq, suffix, strict=True):
    """
    Get entity according to B/I/O-EntityClass Scheme
    :param tag_seq: Iterable tag sequence
    :param char_seq: Iterable char sequence
    :param suffix: EntityClass in (B/I)-EntityClass
    :param strict: bool. If true, raise ValueError when tag_seq is invalid.
    :return:
    """
    entities = []
    i_tag, b_tag = 'I-%s' % suffix, 'B-%s' % suffix
    entity_name = None
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == b_tag:
            if entity_name is not None:
                entities.append(entity_name)
            entity_name = char
        elif tag == i_tag:
            if entity_name is not None:
                entity_name += char
            elif strict:  # I-EntityClass did not come after B-EntityClass
                raise ValueError("Sequence does not comply with BIO scheme.")
        else:
            if entity_name is not None:
                entities.append(entity_name)
                entity_name = None
    if entity_name is not None:
        entities.append(entity_name)
    return entities


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
