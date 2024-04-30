import re

import pysrt
from pysrt import SubRipItem

from const import dash_pattern

MAX_SUB_LENGTH = 60
MIN_SUB_LENGHT = 10
MAX_LINE_LENGTH = MAX_SUB_LENGTH / 2

sent_pattern = r"[!.?]"
pattern = r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"""


class ToSubs:
    def __init__(self, times, lang):
        self.times = times
        self.last_idx = len(times) - 1
        self.is_heb = lang == "he"

    def run(self):
        res_file = pysrt.SubRipFile()
        idx = 0
        sub_idx = 0
        while not idx == self.last_idx:
            (n_idx, prior) = self.find_next_idx_by_symbol(idx)
            if prior < 2:
                (n_idx, pause) = self.find_next_idx_by_pause(idx)
                if pause == 0:
                    n_idx = self.find_next_idx_by_len(idx)
            if idx == n_idx:
                n_idx += 1
            n_idx = min(n_idx, self.last_idx)

            txt = self.get_sub_text(idx, n_idx)
            start = self.sub_by_idx(idx)['start']
            end = self.patch_end_time(n_idx)
            sub = SubRipItem(sub_idx, pysrt.SubRipTime(seconds=start), pysrt.SubRipTime(seconds=end), txt)
            res_file.append(sub)

            idx = n_idx
            sub_idx += 1
        return res_file

    def find_next_idx_by_symbol(self, start_idx):
        ln = 0
        dash_count = 1
        sent_count = 0
        punct_count = 0
        max_prior = (0, 0)
        idx = start_idx
        while ln < MAX_SUB_LENGTH and self.has_idx(idx):
            w = self.sub_by_idx(idx)['word']
            ln += len(w)
            idx += 1

            if self.is_heb and re.search(dash_pattern, w) is not None and start_idx != idx - 1:
                dash_count += 1
                next_prior = dash_count * 9
                if next_prior > max_prior[1]:
                    max_prior = (idx - 1, dash_count * 9)
            if dash_count == 3:
                max_prior = (idx - 1, 100)
                break

            if ln < MIN_SUB_LENGHT:
                continue
            if self.is_heb and re.search(dash_pattern, w) is not None and ln > MAX_LINE_LENGTH:
                max_prior = (idx - 1, 100)
                break
            if re.search(sent_pattern, w) is not None:
                sent_count += 1
                max_prior = (idx, sent_count * 10)
                continue
            if max_prior[1] < 10 and re.search(pattern, w) is not None:
                punct_count += 1
                max_prior = (idx, punct_count * 2)
                continue
        return max_prior

    def find_next_idx_by_len(self, idx):
        ln = 0
        while ln < MAX_SUB_LENGTH and self.has_idx(idx):
            w = self.sub_by_idx(idx)['word']
            ln += len(w)
            idx += 1
        return idx

    def find_next_idx_by_pause(self, idx):
        ln = 0
        max_pause_idx = (idx, 0)
        while ln < MAX_SUB_LENGTH and self.has_idx(idx):
            w = self.sub_by_idx(idx)['word']
            ln += len(w)
            idx += 1
            if ln < MIN_SUB_LENGHT:
                continue

            pause = self.calc_pause(idx)
            if pause >= max_pause_idx[1]:
                max_pause_idx = (idx, pause)

            if ln > MAX_SUB_LENGTH:
                break
        return max_pause_idx

    def calc_pause(self, idx):
        # for last item
        if not self.has_idx(idx + 1):
            return 1000
        return self.sub_by_idx(idx + 1)['start'] - self.sub_by_idx(idx)['end']

    def get_sub_text(self, start_idx, end_idx):
        data = []
        offset = 0
        has_dash = False
        for idx in range(start_idx, end_idx):
            w = self.sub_by_idx(idx)['word'].strip()
            _len = len(w)
            if self.is_heb and re.search(dash_pattern, w) and idx != start_idx:
                w = f"\n{w}"
                has_dash = True
            data.append(({"w": w, "o": offset, "l": _len}))
            offset += _len
        if offset < MAX_LINE_LENGTH or has_dash:
            return " ".join(list(map(lambda d: d["w"], data)))

        median = offset / 2
        idx = 0
        diff = 0.5
        for i in range(len(data) - 1):
            if re.search(pattern, data[i]["w"]):
                _diff = abs(data[i]["o"] + data[i]["l"] - median) / median
                if _diff < diff:
                    diff = _diff
                    idx = i
        if idx == 0:
            for i in range(len(data) - 1):
                if data[i]["o"] + data[i]["l"] > median >= data[i]["o"] - 1:
                    idx = i - 1

        data[idx]["w"] = f"{data[idx]['w']}\n"
        return " ".join(list(map(lambda d: f"{d['w']}", data)))

    def sub_by_idx(self, idx):
        # return self.times[f"{idx}"]
        return self.times[idx]

    def has_idx(self, idx):
        # return f"{idx}" in self.times
        return idx in self.times

    def patch_end_time(self, idx):
        end = self.sub_by_idx(idx - 1)['end']
        if not self.has_idx(idx + 1):
            return end
        n_start = self.sub_by_idx(idx)['start']
        return n_start - max(0.1, (n_start - end) / 2)
