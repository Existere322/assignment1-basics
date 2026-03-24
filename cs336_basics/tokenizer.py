import json
import regex
from collections import Counter
from collections import defaultdict
from typing import Iterable, Iterator
import heapq


class tokenizer():

    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_rank = {}
        self.revocab = {}

        vocab_size = max(vocab.keys()) if vocab else 0

        if special_tokens:
            for token in special_tokens:
                vocab[vocab_size] = token.encode('utf-8')
                vocab_size += 1

        rank = 0
        for merge in merges:
            self.merge_rank[merge] = rank
            rank += 1

        for i in vocab:
            self.revocab[vocab[i]] = i
  
    
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        
        return cls(vocab, merges, special_tokens)
    

    @staticmethod
    def _load_vocab(vocab_filepath):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = {int(k): v.encode('utf-8') for k, v in data.items()}
        return vocab
    

    @staticmethod
    def _load_merges(merge_filepath):
        merges = []
        with open(merge_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts==2):
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))

        return merges
    

    def encode(self, text: str) -> list[int]:

        pre_tokens = Counter() # token_count
        has_special_token = False

        # 1. pre-tokenize the sequence and protect the special tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [regex.escape(tok) for tok in sorted_tokens]
            pattern = "(" + "|".join(escaped_tokens) + ")"
            split_text = regex.split(pattern, text)
            split_special_tokens = split_text[1::2]
            split_text = split_text[::2]
            has_special_token = True
        else:
            split_text = [text]

        # 2. now we should get the split-words
        # List of pre-tokenized chunks, each chunk is a list of words (str) split by PAT
        word_lists = []
        
        for text in split_text:
            words = regex.findall(pattern=PAT, string=text)
            word_lists.append(words)
            pre_tokens.update(words)

        strtoken_to_IDs = {}

        # 3. merge the word by the order of the merges
        for token in pre_tokens:
            str_token = token
            token = [bytes([b]) for b in token.encode('utf-8')]
            merge_token = []
            # Map each adjacent pair to its starting index for heap initialization
            # Use arrays to simulate a doubly linked list for O(1) sequence updates
            n = len(token)
            pre = list(range(-1, n - 1))    # [-1, 0, 1, ..., n-2]
            next = list(range(1, n)) + [-1] # [1, 2, ..., n-1, -1]
            pair_valid = {}
            
            pair_heap = []
            # use priority queue to store the pairs and its rank
            # each time pop out the first rank pair to merge 
            # and then push the new pair
            for i in range(n - 1):
                if((token[i], token[i + 1]) in self.merge_rank):
                    heapq.heappush(pair_heap, (self.merge_rank[(token[i], token[i + 1])], i))
                    pair_valid[(self.merge_rank[(token[i], token[i + 1])], i)] = True

            while pair_heap:
                rank, start_index = heapq.heappop(pair_heap)

                if not pair_valid[rank, start_index]:
                    continue
                # compute the two bytes in this pair
                s1 = start_index
                s2 = next[start_index]
                if s2 == -1: continue
                pair1 = b''
                pair2 = b''
                while s1 < s2:
                    pair1 += token[s1]
                    s1 += 1
                end = n if next[s2] == -1 else next[s2]
                while s2 < end:
                    pair2 += token[s2]
                    s2 += 1
                merge_pair = (pair1, pair2)
                
                # make sure the pair is valid
                if(pair_valid[(self.merge_rank[merge_pair], start_index)]):
                    pair_valid[(self.merge_rank[merge_pair], start_index)] = False
                    pair1_i = start_index
                    pair2_i = next[start_index]

                    # update the next/pre pointer
                    if next[pair2_i] != -1: 
                        next[pair1_i] = next[pair2_i]
                        pre[next[pair2_i]] = pair1_i
                    else:
                        next[pair1_i] = -1

                    next[pair2_i] = -1
                    pre[pair2_i] = -1
                    
                    s1 = pre[pair1_i]
                    s2 = next[pair1_i]
                    new_bytes = pair1 + pair2

                    if(s1 != -1):
                        i = s1
                        pre_bytes = b''
                        while i < pair1_i:
                            pre_bytes += token[i]
                            i += 1
                        if (pre_bytes, pair1) in self.merge_rank:
                            pair_valid[(self.merge_rank[(pre_bytes, pair1)], s1)] = False
                        if (pre_bytes, new_bytes) in self.merge_rank:
                            pair_valid[(self.merge_rank[(pre_bytes, new_bytes)], s1)] = True
                            heapq.heappush(pair_heap, (self.merge_rank[(pre_bytes, new_bytes)], s1))
                    
                    if(s2 != -1):
                        i = s2
                        end = n if next[s2] == -1 else next[s2]
                        next_bytes = b''
                        while i < end:
                            next_bytes += token[i]
                            i += 1
                        if (pair2, next_bytes) in self.merge_rank:
                            pair_valid[(self.merge_rank[(pair2, next_bytes)], pair2_i)] = False
                        if (new_bytes, next_bytes) in self.merge_rank:
                            pair_valid[(self.merge_rank[(new_bytes, next_bytes)], pair1_i)] = True
                            heapq.heappush(pair_heap, (self.merge_rank[(new_bytes, next_bytes)], pair1_i))
            start = 0
            keep_on = True
            while keep_on:
                end = next[start]
                if end == -1:
                    end = n
                    keep_on = False
                bytes_i = b''
                while start < end:
                    bytes_i += token[start]
                    start += 1
                merge_token.append(bytes_i)

            token_IDs = []

            for part in merge_token:
                token_IDs.append(self.revocab[part])

            strtoken_to_IDs[str_token] = tuple(token_IDs)

        # link the special_token IDs with the text IDs
        i = 0
        j = 0
        if has_special_token:
            len_special = len(split_special_tokens)
        len_text = len(word_lists)
        turn = True
        result = []

        while has_special_token and i < len_text and j < len_special:
            if turn:
                if word_lists[i]:
                    for k in range(len(word_lists[i])):
                        for num in strtoken_to_IDs[word_lists[i][k]]:
                            result.append(num)
                i += 1
            else:
                result.append(self.revocab[split_special_tokens[j].encode('utf-8')])
                j += 1

            turn = not turn

        while i < len_text:
            if word_lists[i]:
                for k in range(len(word_lists[i])):
                    for num in strtoken_to_IDs[word_lists[i][k]]:
                        result.append(num)
            i += 1

        while has_special_token and j < len_special:
            result.append(self.revocab[split_special_tokens[j].encode('utf-8')])
            j += 1

        return result

        # merge the bytes in the order of merges
        # Map each token (byte sequence) to its corresponding ID in the vocabulary


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        chunk_size = 1024 * 1024  # 1MB
        pre_pos = []

        # 构建安全边界的查找 pattern
        if self.special_tokens:
            escaped = [regex.escape(tok) for tok in self.special_tokens]
            boundary_pattern = "(" + "|".join(escaped) + r"|\n| )"
        else:
            boundary_pattern = r"(\n)"

        for trunk in iterable:
            buffer += trunk 

            while len(buffer) >= chunk_size:
                search_area = buffer[:chunk_size]
                matches = list(regex.finditer(boundary_pattern, search_area))

                if matches:
                    cut_pos = matches[-1].end()
                else:
                    raise IndexError
                
                pre_pos = buffer[:cut_pos]
                buffer = buffer[cut_pos:]

                # 因为整个输入序列很大，我们应该处理一部分就返回一部分结果，
                # 而不是全部处理完再返回，因此使用 yield from 函数进行处理
                yield from self.encode(pre_pos)

        if buffer:
            yield from self.encode(buffer)

    
    def decode(self, ids: list[int]) -> str:

        result = b''
        for i in ids:
            if i in self.vocab:
                result += self.vocab[i]
            else:
                raise ValueError

        return result.decode('utf-8', errors='replace')





        
    

