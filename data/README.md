# Data Sources and Licenses

All baseline ("ideal") clips are trimmed excerpts of public-domain recordings,
converted to 16 kHz mono 16-bit WAV. No other modification is applied to the
baseline files. Flawed versions are generated deterministically via
`python -m dataset_builder.build_dataset`.

The set mixes two original archival speeches by the speakers themselves
(Kennedy, Eisenhower) and two volunteer interpretive readings (FDR text,
Lincoln text), covering both formal oratory and reading styles.

| Clip ID | Speaker / Work | Date | Excerpt (source file time) | Source |
|---|---|---|---|---|
| `kennedy_inaugural` | John F. Kennedy, Inaugural Address | 20 Jan 1961 | 00:01:53–00:03:23 | [Miller Center](https://millercenter.org/the-presidency/presidential-speeches/january-20-1961-inaugural-address) |
| `eisenhower_farewell` | Dwight D. Eisenhower, Farewell Address | 17 Jan 1961 | 00:01:18–00:02:46 | [Miller Center](https://millercenter.org/the-presidency/presidential-speeches/january-17-1961-farewell-address) |
| `fdr_fireside_01` | Franklin D. Roosevelt, Fireside Chat 1 (text), read by a LibriVox volunteer | 12 Mar 1933 (text) | 00:00:26–00:01:55 | [Internet Archive (LibriVox collection)](https://archive.org/details/firesidechats_1705_librivox) |
| `lincoln_gettysburg` | Abraham Lincoln, Gettysburg Address (text), read by a LibriVox volunteer | 19 Nov 1863 (text) | 00:00:08–00:01:53 | [LibriVox](https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/) |

## License and Status

| Clip ID | Status | Basis |
|---|---|---|
| `kennedy_inaugural`, `eisenhower_farewell` | Public domain (US federal government work) | Speeches delivered by the President in an official capacity (17 U.S.C. section 105). Audio hosted by the Miller Center from presidential library holdings. |
| `fdr_fireside_01` | Public domain | The text is FDR's address, a US federal work. The recording is a LibriVox volunteer reading released into the public domain. |
| `lincoln_gettysburg` | Public domain | Text published 1863. The recording is a LibriVox volunteer reading released into the public domain. |

## Own Recordings (Real, Human-Performed Validation Sets)

- `data/own_recordings/` holds three takes of the Lincoln transcript recorded by the project author: `own_good` (clean delivery) and `own_flawed_moderate` / `own_flawed_subtle` (flaws performed intentionally at four planned sentences: pace, monotone, pause, volume).
- `data/own_recordings_b/` holds an untouched held-out clean take (`ownb_good.wav`) of the Kennedy transcript recorded by the project author to evaluate cross-speaker false-positive rates on unseen text.
- Released under CC BY 4.0.