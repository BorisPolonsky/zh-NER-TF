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


def get_BIO_entity_boundaries(tag_seq, char_seq, suffixes, strict=True):
    """
    Get entity according to B/I/O-EntityClass Scheme
    :param tag_seq: Iterable tag sequence
    :param char_seq: Iterable char sequence
    :param suffixes: A list of EntityClass in (B/I)-EntityClass
    :param strict: bool. If true, raise ValueError when tag_seq is invalid.
    :return: a tuple of detected entities from left to right.
    """
    entities = []
    entity = [None] * 3
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        token, suffix = tag[:2], tag[2:]
        if suffix in suffixes:
            if token == "B-":
                if entity[2] is not None:
                    entities.append(tuple(entity))
                entity = [i, i + 1, suffix]
                continue
            elif token == "I-":
                if entity[2] != suffix:  # I-EntityClass did not come after B-EntityClass
                    if strict:
                        raise ValueError("Sequence does not comply with BIO scheme.")
                    else:
                        if entity[2] is not None:
                            entities.append(tuple(entity))
                        entity = [i, i + 1, suffix]
                else:
                    entity[1] = i + 1
                continue
        if entity[2] is not None:
            entities.append(tuple(entity))
            entity = [None] * 3
    if entity[2] is not None:
        entities.append(tuple(entity))
    return entities


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


def get_logger(filename, logger_name=None):
    """
    The function will return the created logger only
    without adding additional handlers to it.
    In addition, calling this function will configure root logger,
    if and only if it's not yet done before.
    :param filename: Path for FileHandler of root logger.
    :param logger_name: Name of the logger.
    :return: logger
    """
    fmt = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.DEBUG)
    # The following state does nothing if the root logger is already configured.
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    return logger
