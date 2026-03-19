import os
from typing import BinaryIO
from itertools import repeat
import regex
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from collections import Counter
import heapq


def vocab_init(
    special_tokens: list[str]
) -> dict[int, bytes]:
    # Python 中 dict 初始化为 {} 列表初始化为 []
    # 字典初始化后可以直接赋值，但不能直接 += 1 这样子
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])
        # bytes([i]) 返回 i 对应的字节类型的内容
        # [1] 是一个列表，在 python 中列表是一种可迭代对象
    
    start = 256
    for token in special_tokens:
        vocab[start] = token.encode('utf-8')
        start = start + 1

    return vocab


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(
    input_path: str | os.PathLike, 
    start: int, 
    end: int, 
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    with open(input_path, 'rb') as f:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 对句子进行分词的正则表达式

        word_count = Counter()
        word_bytes_count = Counter()

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        
        if special_tokens:
            escaped_tokens = [regex.escape(tok) for tok in special_tokens]
            pattern = "|".join(escaped_tokens)
            split_chunk = regex.split(pattern, chunk)
        else:
            split_chunk = [chunk]

        for part in split_chunk: 
            word_bytes_count.update(regex.findall(pattern=PAT, string=part))
            # 注意两种方法的区别，update 需要接收到可遍历的东西，比如列表
            # for match in regex.finditer(PAT, part):
            #     word_str = match.group()
            #     word_bytes_count[word_str] += 1
        

        # 在之前得到的是字符串形式的内容，需要将字符串 encode 为字节流的格式，然后通过遍历得到字节流元组
        for word_bytes, count in word_bytes_count.items():
            word_count[tuple(bytes([b]) for b in word_bytes.encode('utf-8'))] = count

        return word_count


def pre_tokenization(
    input_path: str | os.PathLike, 
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    
    # 在这一步骤中，调用预分词的分块代码进行分块
    # 分块后将不同的分块部署到不同的线程上
    input_paths = []
    starts = []
    ends = []
    special_tokenss = []
    global_counts = Counter()

    with open(input_path, "rb") as f:
        num_processes = 16
        # file_size = f.tell()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # # 关键优化1：如果实际只有一个 chunk，直接串行，避免进程启动开销
        # ranges = list(zip(boundaries[:-1], boundaries[1:]))
        # if len(ranges) <= 1:
        #     if ranges:
        #         s, e = ranges[0]
        #         return process_chunk(input_path, s, e, special_tokens)
        #     return global_counts

        # # 关键优化2：小文件直接串行通常更快（这个阈值可按机器调）
        # if file_size < 1_000_000:
        #     for s, e in ranges:
        #         global_counts.update(process_chunk(input_path, s, e, special_tokens))
        #     return global_counts
        
        
        starts = boundaries[:-1]
        ends = boundaries[1:]

        # 先尝试单线程逻辑
        # result = process_chunk(input_path, start, end, special_tokens)
        # global_counts.update(result)
        max_workers = 16
        if len(starts) < max_workers: 
            max_workers = len(starts)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_chunk, 
                                    repeat(input_path), 
                                    starts, 
                                    ends, 
                                    repeat(special_tokens), )
            
            for local_counts in results:
                global_counts.update(local_counts)
        # 可以通过 update 对字典进行更新

    # 先对单线程进行调试

    
    
    return global_counts


class LargePair:
    """包装类：用于在小根堆中实现大根堆的效果"""
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        # 核心比较逻辑：
        # 1. 首先比较次数 (count越大，认为对象越“小”，越容易排在堆顶)
        if self.count != other.count:
            return self.count > other.count
        # 2. 次数相同时，比较字典序 (pair大，认为对象越“小”，越容易排在堆顶)
        return self.pair > other.pair

    
    """
    计算步骤
    1. 首先要找到每一对的出现次数，并且统计起来
    2. 其次需要找到出现次数最多的一对
    3. 之后是更新涉及到这一对的单词的次数
    4. 更新完涉及到这一对的单词之后
    e st, wa st -> est, wa st 当一个邻接对合并以后计算次数就不再有他们了，将仅仅计算合并后的次数
    比如 w e s t, e a s t, n i c e s t, 首先合并 st, 之后变为 w e st, e a st, n i c e st
    之后再合并 est, 得到 w est, e a st, n i c est 这样的结果
    5. 每次合并，上次统计的邻接对次数就作废了，因为我们如果合并了 st 那么邻接对 e s 如果后面有 t 就会作废
    而我们并不知道哪些与 st 邻接，因此无法改变它们的次数？

    如果我们在邻接对上做文章，需要两件事，首先是去掉合并后的邻接对的次数
    其次找出与这次合并有关的单词，将邻接对合并，并且更新与合并后内容相关联的邻接对的次数
    同时加入与合并后内容作为邻接对的新邻接对的次数

    我们保存了每个邻接对所对应的 word ，但是每个邻接对所对应的 word 并不完全正确，
    """ 

def merge(
    pre_tokens: dict[tuple[bytes], int], 
    vocab_size: int, 
    vocab: dict[int, bytes]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab_num = len(vocab)
    pairs = Counter()                   # 每个 pair 出现的次数
    pair_token = defaultdict(set)       # 每个 pair 对应的 pre-token 索引
    new_pre_token = {}                  # 用每个 pre-token 的索引存储其出现次数
    words = []                          # pre-token 的索引与内容
    merge_list = []                 


    # 首先计算邻接对的出现次数，以及每个邻接对所在的单词
    # 对于每个邻接对对应的单词，我们通过索引的方式指向一个列表
    # 这样可以更新该列表而不必改变整个索引
    # TODO: 这部分可以通过并行化进行加速，只需要分成四块统计即可
    index = 0
    for token in pre_tokens:
        words.append(token)
        new_pre_token[index] = pre_tokens[token]
        for pre, back in zip(token[:-1], token[1:]):
            pairs[(pre, back)] += pre_tokens[token]
            pair_token[(pre, back)].add(index)
        index += 1

    # 由于后面的合并，每个单独的单词都会发生改变，因此 pre_tokens 也应该存储 words 的序号而不是 tuple 本身
            
    # 构建优先队列保存 pair 的数量
    pair_heap = [LargePair(count, pair) for pair, count in pairs.items()]

    heapq.heapify(pair_heap)


    # 我们首先保存每个 pair 对应的 word 有哪些
    # 然后根据合并的 pair 对 word 涉及到的 pair 次数进行更新
    # 采用 dict 存储每个 pair 的次数
    # 同时采用惰性更新的大根堆计算最大的次数，取堆顶与 pair dict 比较，一致说明数据有效则进行更新，否则舍弃
    while vocab_num < vocab_size : 
        # 从堆中获得次数最多的邻接对
        # 每次从队列取出元素之前先判空
        if not pair_heap :
            return (vocab, merge_list)
        tuple_top_pair = heapq.heappop(pair_heap)

        # 取出邻接对之后先判断是否有效
        while tuple_top_pair.count != pairs[tuple_top_pair.pair]:
            if not pair_heap :
                return (vocab, merge_list)
            tuple_top_pair = heapq.heappop(pair_heap)
        
        top_pair = tuple_top_pair.pair

        num_changed = set()

        # 遍历含有该邻接对的 token 进行更新
        for num in pair_token[top_pair]:
            word = words[num]
            new_word = []
            combine_pl = []
            insert_pl = []
            pre = False

            # TODO: 需要用正确的方式去查找存在的邻接对，然后再进行更新

            # 我们只修改合并的 pair 前后，pre, pair1, pair2, back (pair = pair1 + pair2)
            # pre, pair1 和 pair2, back 首先在 pair_token 中去掉 word 这个词对应的 num，其次 pairs 上减少次数 new_pre_token[num]
            # pre, pair 和 pair, back 在 pair_token 中加上 num，且 pairs 上次数加上 new_pre_token[num]
            
            # 首先找出原来词的位置
            i = 0
            j = 0
            while i < len(word): 
                if i + 1 < len(word) and (word[i], word[i + 1]) == top_pair:
                    combine_pl.append(i)
                    insert_pl.append(j)
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                    pairs[top_pair] -= new_pre_token[num]
                else:
                    new_word.append(word[i])
                    i += 1
                j += 1

            new_word = tuple(new_word)
            words[num] = new_word

            # combine_pl 代表合并的位置，word[i-1],word[i] 和 word[i+1],word[i+2] 都是需要删去减少的
            # insert_pl 代表合并后内容在新词中的位置，new_word[j-1], new_word[j] 和 new_word[j], new_word[j+1] 都是需要插入增加的
            # 两者都需判定，若前面是合并/新插入内容，则不进行增加
            # 问题：a b c d a b 若合并 b c 会导致 a b 被移除，

            # 之后减去原来词所有邻接对的次数
            for i in range(len(combine_pl)):
                pl = combine_pl[i]
                
                # 首先判断 i 前一个是不是 pl - 2 对应的位置，若是不做改变
                if pl-1 >= 0 and ((i-1 >= 0 and (pl-2) != combine_pl[i-1]) or (i-1 < 0)):
                    pre_tuple = (word[pl-1], word[pl])
                    if pl + 2 < len(word) and word[pl] == b' ' and word[pl+1] == b't' and word[pl+2] == b'h':
                        print("the right place.")
                    pairs[pre_tuple] -= new_pre_token[num]
                    pair_token[pre_tuple].discard(num)
                    num_changed.add(pre_tuple)
                
                # 如果有后一个邻接对，则同样修改
                if pl+2 < len(word) and top_pair != (word[pl+1], word[pl+2]): 
                    back_tuple = (word[pl+1], word[pl+2])
                    pairs[back_tuple] -= new_pre_token[num]
                    pair_token[back_tuple].discard(num)
                    num_changed.add(back_tuple)

            for j in range(len(insert_pl)):
                pl = insert_pl[j]

                # 首先判断 j 前一个内容是不是 pl - 1 对应的位置，若是不改变
                if pl-1 >= 0 and ((j-1 >= 0 and (pl-1) != insert_pl[j-1]) or j-1 < 0):
                    pre_tuple = (new_word[pl-1], new_word[pl])
                    pairs[pre_tuple] += new_pre_token[num]
                    pair_token[pre_tuple].add(num)
                    num_changed.add(pre_tuple)

                if pl+1 < len(new_word):
                    back_tuple = (new_word[pl], new_word[pl+1])
                    pairs[back_tuple] += new_pre_token[num]
                    pair_token[back_tuple].add(num)
                    num_changed.add(back_tuple)

            for pre, back in zip(new_word[:-1], new_word[1:]):
                pair_token[(pre, back)].add(num)


        for pair in num_changed:
            heapq.heappush(pair_heap, LargePair(pairs[pair], pair))


        combine = b"".join(top_pair)
        vocab[vocab_num] = combine
        vocab_num += 1
        merge_list.append(top_pair)
  
    return (vocab, merge_list)






