import logging
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertForQuestionAnswering


def lab7():
    try:
        """Lab 7: NLP with text by using BERT model."""
        logging.info("LAB7. PREPARATION")
        logging.info("loading BERT model and tokenizer")
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        text_1 = "I understand equations, both the simple and quadratical."
        text_2 = "What kind of equations do I understand?"
        indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
        logging.info("our sentences to tokenize \nT1: %s \nT2: %s", text_1, text_2)
        logging.info("tokenized and indexed tokens: %s", indexed_tokens)
        logging.info("converted tokens back to text to see what used as tokens: %s", tokenizer.convert_ids_to_tokens([str(token) for token in indexed_tokens]))
        logging.info("We have more tokens/words because many languages have word roots, or components that make up a word. "
                     "\nFor instance, the word 'quadratic' has the root 'quadr' which means '4' ")
        logging.info("looking for segment_ids for BART")
        # BERT need to know segment_ids. We can use `special_tokens` added by `tokenizer` to fine them
        segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)
        logging.info("Text Masking")
        masked_index = 5
        indexed_tokens[masked_index] = tokenizer.mask_token_id
        tokens_tensor = torch.tensor([indexed_tokens])
        logging.info("applied test masking result: %s", tokenizer.decode(indexed_tokens))
        logging.info("loading model than used to predict missing words")
        masked_lm_model = BertForMaskedLM.from_pretrained("bert-base-cased")
        with torch.no_grad():
            predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)
        logging.info("prediction model test result: %s", predictions)
        # since it is hard to read - we can use shape and tokens to find a word
        predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        logging.info("predicted token by model: %s", predicted_token)

        logging.info("example of another BERT version with own tokenizer")
        question_answering_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
        segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)
        question_answering_model = BertForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")
        with torch.no_grad():
            out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)
        answer_sequence = indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits) + 1]
        logging.info("predicted tokens: %s", answer_sequence)
        logging.info("decoded prediction words by tokens: %s", question_answering_tokenizer.convert_ids_to_tokens(answer_sequence))
        logging.info("decoded prediction by tokens sum as single word: %s", question_answering_tokenizer.decode(answer_sequence))

    except Exception as e:
        logging.error("Error in lab7: %s", e)
        raise


def get_segment_ids(indexed_tokens, sep_token=102):
    logging.info("Next, we can create a for loop. We'll start with our segment_id set to 0, "
                 "\nand we'll increment the segment_id whenever we see the [SEP] token. "
                 "\nFor good measure, we will return both the segment_ids and indexd_tokens as "
                 "\ntensors as we will be feeding these into the model later.")
    segment_ids = []
    segment_id = 0
    for token in indexed_tokens:
        if token == sep_token:
            segment_id += 1
        segment_ids.append(segment_id)
    segment_ids[-1] -= 1  # Last [SEP] is ignored
    return torch.tensor([segment_ids]), torch.tensor([indexed_tokens])
