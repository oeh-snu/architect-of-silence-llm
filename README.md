# Architect of Silence: Persona-prompted Large Language Models Make Minority Population and Social Suffering Invisible
Soomin Kim, Eun-Ji Paeng, Seung-Sup Kim<br>
Graduate School of Public Health, Seoul National University<br>

English | [한국어](#한국어)

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [Samples](#samples)
- [System message](#system-message)
- [Not in this repository](#not-in-this-repository)
- [License](#license)

## Overview
This repository holds the materials needed to reconstruct the inputs of the
study: which personas were used, and how each persona was given to the model.

```
Nemotron-Personas-Korea (1,000,000 personas)
      ↓  simple random sample without replacement
10,000 personas (main sample)  ──→  1,000 personas (nested sub-sample)
Llama 3.3 70B Instruct              GPT-6 Astra, Claude Opus 5, Gemini 3.8 Flash
      ↓                                   ↓
one conversation per persona and model: system message, then survey questions
      ↓
survey responses, compared with human responses
from nationally representative surveys
```

Each persona was written into the system message, and the survey questions were
then asked as consecutive user messages of the same conversation. The main
sample was run on Llama 3.3 70B Instruct at temperature 0, with each response
constrained to a JSON schema. The sub-sample was also run on GPT-6 Astra,
Claude Opus 5 and Gemini 3.8 Flash for cross-model comparison. The responses
were compared with human responses from nationally representative surveys.

## Files

| File | Contents |
|---|---|
| `npk_sample_10k.uuids.txt` | `uuid`s of the 10,000 personas in the main sample |
| `npk_sample_1k.uuids.txt` | `uuid`s of the 1,000 personas in the sub-sample |
| `system_message.txt` | System message template that assigns a persona to the model |

## Samples
Personas are from
[Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)
(NPK), revision `ada0f5b53a38bb5a30cce09358adde883c1ab63a` (June 22, 2026),
`train` split. All 1,000,000 personas (ages 19–99) were eligible.

| File | n | Drawn from | SHA-256 | Model |
|---|---|---|---|---|
| `npk_sample_10k.uuids.txt` | 10,000 | NPK, 1,000,000 personas | `9463e5198c8cdbbe2828492af3decb43ea66d79bddb1ecae0369044f93477b96` | Llama 3.3 70B Instruct |
| `npk_sample_1k.uuids.txt` | 1,000 | `npk_sample_10k` | `8340706122d1cd8b6727146d9a944778f6268c4615f712f6e077137a7bb80995` | GPT-6 Astra, Claude Opus 5, Gemini 3.8 Flash |

`npk_sample_10k` is the main sample, a simple random sample without
replacement. `npk_sample_1k` is a simple random sub-sample nested within it,
used for the cross-model comparison; every persona in it is also in the main
sample, so its Llama 3.3 responses can be paired with those of the other models
by `uuid`.

Each file has one `uuid` per line, in ascending order, with LF line endings and
no newline after the last line. In this format the SHA-256 of each file is the
digest above:

```sh
shasum -a 256 npk_sample_10k.uuids.txt npk_sample_1k.uuids.txt
```

To retrieve the personas:

```python
from datasets import load_dataset

ids = set(open("npk_sample_10k.uuids.txt").read().split("\n"))
npk = load_dataset(
    "nvidia/Nemotron-Personas-Korea",
    revision="ada0f5b53a38bb5a30cce09358adde883c1ab63a",
    split="train",
)
sample = npk.filter(lambda u: u in ids, input_columns="uuid")
assert len(sample) == len(ids)
```

## System message
`system_message.txt` is the system message sent for every persona. It
is in Korean.

- `{...}` marks where a persona's field is inserted. Each name is the NPK
  column of the same name. All 25 NPK columns other than `uuid` are used.
- The first block is the 12 demographic fields on one line, separated by ` | `.
  The 13 narrative fields follow, each under its `[column name]` heading.
- `skills_and_expertise_list` and `hobbies_and_interests_list` are lists in
  NPK. Each element is written on its own line, prefixed with `- `.
- A field that is empty for a persona is left out together with its label or
  heading.
- The first and last lines are fixed. They tell the model that it is the person
  described, living in Korea, and ask it to answer the survey questions that
  follow honestly as that person.

## Not in this repository
- **Persona content.** Only identifiers are listed here. Retrieve the personas
  from NPK as shown above.
- **Survey questions.** Majority of the questions were adopted from the
  [Korea National Health and Nutrition Examination Survey](https://knhanes.kdca.go.kr/knhanes/eng/main.do)
  (KNHANES), the
  [Korea Welfare Panel Study](https://www.koweps.re.kr:442/eng/main.do)
  (KOWEPS) and the
  [Korea Health Panel Study](https://www.khp.re.kr:444/eng/main.do) (KHPS).
  Further details are listed in Supplementary Table 2 of the paper.
- **Human responses.** The human responses that the model responses were
  compared with come from these nationally representative surveys. Their data
  are publicly available from the survey websites linked above.

## License
The files in this repository are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `LICENSE`.

Nemotron-Personas-Korea is published by NVIDIA under CC BY 4.0.

---

# 한국어

[English](#architect-of-silence-persona-prompted-large-language-models-make-minority-population-and-social-suffering-invisible) | 한국어

## 목차
- [개요](#개요)
- [파일](#파일)
- [표본](#표본)
- [시스템 메시지](#시스템-메시지)
- [이 저장소에 없는 것](#이-저장소에-없는-것)
- [라이선스](#라이선스)

## 개요
이 저장소는 연구의 입력을 재구성하는 데 필요한 자료를 담고 있습니다. 어떤
페르소나를 사용했는지, 그리고 각 페르소나를 모델에 어떻게 제시했는지입니다.

```
Nemotron-Personas-Korea (페르소나 1,000,000개)
      ↓  비복원 단순무작위추출
10,000개 (주 표본)  ──→  1,000개 (주 표본 안의 하위 표본)
Llama 3.3 70B Instruct   GPT-6 Astra, Claude Opus 5, Gemini 3.8 Flash
      ↓                        ↓
페르소나·모델마다 대화 1건: 시스템 메시지, 이어서 설문 문항
      ↓
설문 응답을 전국 대표성이 있는 조사의 사람 응답과 비교
```

각 페르소나를 시스템 메시지에 넣고, 같은 대화 안에서 설문 문항을 연속된
사용자 메시지로 물었습니다. 주 표본은 Llama 3.3 70B Instruct에서 temperature
0으로 실행했으며, 각 응답은 JSON 스키마로 형식을 제한했습니다. 하위 표본은
모델 간 비교를 위해 GPT-6 Astra, Claude Opus 5, Gemini 3.8 Flash에서도
실행했습니다. 모델의 응답은 전국 대표성이 있는 조사의 사람 응답과
비교했습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `npk_sample_10k.uuids.txt` | 주 표본 페르소나 10,000개의 `uuid` |
| `npk_sample_1k.uuids.txt` | 하위 표본 페르소나 1,000개의 `uuid` |
| `system_message.txt` | 모델에 페르소나를 부여하는 시스템 메시지 템플릿 |

## 표본
페르소나의 출처는
[Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)
(NPK)의 리비전 `ada0f5b53a38bb5a30cce09358adde883c1ab63a` (2026년 6월 22일),
`train` 분할입니다. 페르소나 1,000,000개 전체(19–99세)가 추출 대상이었습니다.

| 파일 | n | 추출 모집단 | SHA-256 | 모델 |
|---|---|---|---|---|
| `npk_sample_10k.uuids.txt` | 10,000 | NPK, 페르소나 1,000,000개 | `9463e5198c8cdbbe2828492af3decb43ea66d79bddb1ecae0369044f93477b96` | Llama 3.3 70B Instruct |
| `npk_sample_1k.uuids.txt` | 1,000 | `npk_sample_10k` | `8340706122d1cd8b6727146d9a944778f6268c4615f712f6e077137a7bb80995` | GPT-6 Astra, Claude Opus 5, Gemini 3.8 Flash |

`npk_sample_10k`는 주 표본으로, 비복원 단순무작위표본입니다.
`npk_sample_1k`는 그 안에서 다시 단순무작위추출한 하위 표본으로, 모델 간
비교에 사용했습니다. 하위 표본의 모든 페르소나는 주 표본에도 포함되어 있어
Llama 3.3의 응답과 다른 모델의 응답을 `uuid`로 짝지을 수 있습니다.

각 파일은 한 줄에 `uuid` 하나씩, 오름차순으로, LF 줄바꿈을 사용하며 마지막 줄
뒤에는 줄바꿈이 없습니다. 이 형식에서 각 파일의 SHA-256이 위 표의 값입니다.

```sh
shasum -a 256 npk_sample_10k.uuids.txt npk_sample_1k.uuids.txt
```

페르소나를 불러오는 방법:

```python
from datasets import load_dataset

ids = set(open("npk_sample_10k.uuids.txt").read().split("\n"))
npk = load_dataset(
    "nvidia/Nemotron-Personas-Korea",
    revision="ada0f5b53a38bb5a30cce09358adde883c1ab63a",
    split="train",
)
sample = npk.filter(lambda u: u in ids, input_columns="uuid")
assert len(sample) == len(ids)
```

## 시스템 메시지
`system_message.txt`는 모든 페르소나에 대해 보낸 시스템 메시지이며,
한국어로 작성되어 있습니다.

- `{...}`는 페르소나의 필드가 들어가는 자리입니다. 각 이름은 같은 이름의 NPK
  열입니다. `uuid`를 제외한 NPK의 25개 열을 모두 사용합니다.
- 첫 블록은 인구학적 필드 12개를 ` | `로 구분해 한 줄에 적은 것입니다. 이어서
  서술형 필드 13개가 각각 `[열 이름]` 제목 아래에 옵니다.
- `skills_and_expertise_list`와 `hobbies_and_interests_list`는 NPK에서
  리스트입니다. 원소 하나를 한 줄에, 앞에 `- `를 붙여 적습니다.
- 어떤 페르소나에서 값이 비어 있는 필드는 그 라벨이나 제목과 함께 생략됩니다.
- 첫 줄과 마지막 줄은 고정되어 있습니다. 모델에게 자신이 한국에 살고 있는
  해당 인물임을 알리고, 그 인물로서 이어지는 설문 문항에 솔직하게 답하도록
  요청합니다.

## 이 저장소에 없는 것
- **페르소나 내용.** 여기에는 식별자만 있습니다. 페르소나는 위의 방법으로
  NPK에서 불러오십시오.
- **설문 문항.** 대부분의 설문 문항은
  [국민건강영양조사](https://knhanes.kdca.go.kr/knhanes/main.do)(KNHANES),
  [한국복지패널](https://www.koweps.re.kr:442/main.do)(KOWEPS),
  [한국의료패널](https://www.khp.re.kr:444/)(KHPS)에서 가져온 것이며, 자세한
  내용은 논문의 Supplementary Table 2에 제시되어 있습니다.
- **사람 응답.** 모델의 응답과 비교한 사람 응답은 전국 대표성이 있는 위
  조사들에서 가져왔습니다. 이 조사들의 자료는 위에 링크한 각 조사
  홈페이지에서 공개되어 있습니다.

## 라이선스
이 저장소의 파일은
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)으로 공개합니다.
`LICENSE`를 참고하십시오.

Nemotron-Personas-Korea는 NVIDIA가 CC BY 4.0으로 공개한 데이터셋입니다.
