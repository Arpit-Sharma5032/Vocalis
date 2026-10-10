# Data Sources and Licenses

All baseline ("ideal") clips are trimmed excerpts of public-domain recordings,
converted to 16 kHz mono 16-bit WAV. No other modification is applied to the
baseline files. Flawed versions are generated programmatically later (see
`dataset_builder/`).

The set mixes two original speeches by the speakers themselves (Kennedy, Eisenhower) and two volunteer interpretive readings (FDR text, Lincoln text), covering both oratory and reading styles.

| Clip ID | Speaker / Work | Date | Excerpt (source file time) | Source |
|---|---|---|---|---|
| `kennedy_inaugural` | John F. Kennedy, Inaugural Address | 20 Jan 1961 | `00:01:53`-`00:03:23` | [Miller Center](https://millercenter.org/the-presidency/presidential-speeches/january-20-1961-inaugural-address) |
| `eisenhower_farewell` | Dwight D. Eisenhower, Farewell Address | 17 Jan 1961 | `00:01:18`-`00:02:46` | [Miller Center](https://millercenter.org/the-presidency/presidential-speeches/january-17-1961-farewell-address) |
| `fdr_fireside_01` | Franklin D. Roosevelt, Fireside Chat 1 (text), read by a LibriVox volunteer | 12 Mar 1933 (text) | 00:00:26-00:01:55 | [Internet Archive (LibriVox collection)](https://archive.org/details/firesidechats_1705_librivox) |
| `lincoln_gettysburg` | Abraham Lincoln, Gettysburg Address (text), read by a LibriVox volunteer | 19 Nov 1863 (text) | 00:00:08-00:01:53 | [LibriVox](https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/) |

## License and status

| Clip ID | Status | Basis |
|---|---|---|
| `kennedy_inaugural`, `eisenhower_farewell` | Public domain (US federal government work) | Speeches delivered by the President in an official capacity (17 U.S.C. section 105). Audio hosted by the Miller Center, which obtains its media from the presidential libraries. The Miller Center publishes no separate license for these files; status is based on the federal-work rule. |
| `fdr_fireside_01` | Public domain | The text is FDR's address, a US federal work. The recording is a LibriVox volunteer reading, released into the public domain by LibriVox. |
| `lincoln_gettysburg` | Public domain | Text published 1863. The recording is a LibriVox volunteer reading, released into the public domain by LibriVox. |

LibriVox states its recordings are public domain in the United States and not
necessarily in other countries. We are not lawyers; this table records our
good-faith reading of the sources above.

## Transcripts

Transcripts in `data/transcripts/` are excerpts matched word for word to each
audio clip, with numbers spelled out for forced alignment. Word-level
timestamps in `data/processed_metrics/*_alignment.json` were produced with
WhisperX forced alignment, with the supplied transcript as the text.

## Reproducing the clips

```
ffmpeg -y -i <source.mp3> -ss <start> -t <duration_s> -ac 1 -ar 16000 -c:a pcm_s16le data/raw_audio/<clip_id>.wav
```