from pipeline.data_utils.constants import SYSTEM_DETAIL

from .templates import Template

SYSTEM_ACC = "The assistant gives accurate answers to the user's questions."


###############################################################################
# default patterns
###############################################################################

pattern_map = {
    "vqa": ["vqa", "vgqa", "ocrvqa", "okvqa"],
    "vqa-o": ["aokvqa"],  # vqa with options
    "vsr": ["vsr"],
    "kvqa": ["kvqa"],
    "loc": ["vgloc", "refexploc"],
    "captioning": ["coyo100m", "blip", "textcaps"],
}
pattern_dict = {
    # captioning
    "captioning": [
        # (SYSTEM_DETAIL, "Describe this image in detail.", "{caption}"),
        # ("", "Provide a one-sentence caption for the provided image.", "{caption}"),
        ("[NO_PROMPT]", "", "{caption}"),
    ],
    # VQA
    "vqa": [
        ("", "Answer the question using a single word or phrase. {question}", "{answer}"),
    ],
    # GQA -- VQA with answer & full_answer
    "gqa": [
        ("", "Answer the question using a single word or phrase. {question}", "{answer}"),
        # ("", "Answer the following question with a complete sentence. {question}", "{full_answer}"),
    ],
    # VQA with option
    "vqa-o": [
        ("", "Answer with the option's letter from the given choices directly. {question}\nThere are several options:\n{option}\n", "{answer}"),
        # ("", "First provide a rationale for your answer, then select the correct option for the following question. {question}\nThere are several options:\n{option}\n", "Rationale: {rationale}\nAnswer: {answer}"),
    ],
    # SQA has context, lecture, and solution also
    "sqa": [
        ("", "Answer with the option's letter from the given choices directly. {question}\nContext: {context}\nThere are several options:\n{option}\n", "{answer}"),
    ],
    "loc": [
        # visual grounding
        ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        # grounded captioning
        ("", "Provide a short description for this region. {bbox}", "{phrase}"),
    ],
    "vsr": [
        ("", "Answer the question using a single word or phrase. {question_interro} Please answer yes or no.", "{answer}"),
        # original caption with an additional instruction.
        #("", "Answer whether the following caption is correctly describing the given image with yes or no. {question}", "{answer}"),
    ],
    "kvqa": [
        # use original question
        ("", "Answer the question using a single word or phrase. {question}", "{answer}")
        # use paraphrased question
        #("", "Answer the question using a single word or phrase. {paraQuestion}", "{answer}")
    ],

    # REFTEPS-noname
    "refcoco": [
        ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        ("", "Provide a description for the region {bbox}, utilizing positional words to refer to objects. Example: 'The large blue teddy bear next to the red balloon'", "{phrase}"),
    ],
    "refcocop": [
        ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        ("", "Provide a description for the region {bbox}, focusing on the appearance of objects without using positional words. Example: 'The large blue teddy bear holding a red balloon.'", "{phrase}"),
    ],
    "refcocog": [
        ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        ("", "Provide a description for the region {bbox}, using detailed and descriptive expressions to refer to objects. Example: 'The large blue teddy bear holding a red balloon with a joyful expression.'", "{phrase}"),
    ],

    # Evaluation
    "mme": [("", "Answer the question using a single word or phrase. {question}", "")],
    "mmb": [("", "Answer with the option's letter from the given choices directly. {question}", "")],
    #
    "eval-vqa": [("", "Answer the question using a single word or phrase. {question}", "")],
    "eval-sqa": [
        ("", "Answer with the option's letter from the given choices directly. {question}\nContext: {context}\nThere are several options:\n{option}\n", "")
    ],
    "eval-refexploc": [("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "")],
    "eval-vsr": [("", "Answer the question using a single word or phrase. {question_interro} Please answer yes or no.", "")],
}
Template.register("honeybee_default", pattern_dict, pattern_map)
