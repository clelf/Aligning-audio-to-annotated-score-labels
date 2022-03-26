# Aligning-audio-to-annotated-score-labels


**Beat tracking alignment:**
Run pipeline notebook which contains functions and applications to two examples, Bach prelude in C and in Eb.
Consists in tracking beats and downbeats with two algorithms from the _madmom_[https://github.com/CPJKU/madmom] library, extracting the annotated score labels and aligning both beats and score labels sequences to audio data. 

_Beats tracking and labeling - aligning downbeats and beats:_ Downbeats tracking algorithm might track an invalid multiple of correct beats, that the beats algorithm correctly finds. On the other hand, the downbeats algorithm can provide an accurate sense of beats period. The pipeline therefore proposes to combine both algorithms results. Once done, the beats sequence contains the beats timestamps as well as to which beat in a bar and which bar it corresponds.

_Extract the annotated labels sequence from JSON tree:_ annotated labels are identified as the leaves (entities with empty children) of the JSON tree

_Format aligned data to CSV output where each row is an audio sample and the columns are:_
  - timestamp \[timestamp of the audio sample in seconds]
  - LeftChannel \[membrane potential]	
  - RightChannel \[membrane potential]
  - Meter	
  - Bar	\[bar number the audio sample belongs to, 0 if audio sample is located before the first beat]
  - Beat \[beat number within a bar the audio sample belongs t, 0 if audio sample is located before the first beat]	
  - BeatOnsetLocation	\[marks True if the row corresponds to a beat onset, False otherwise]
  - BeatOnsetIndex \[marks the index of the beat within the entire beats sequence when the row is a beat onset row, 0 otherwise]
  - LabelOnset \[marks True if the row corresponds to an annotated label onset, False otherwise]
  - AnnotatedLabel \[marks the label name when the row is a label onset row, 0 otherwise]
  - AnnotatedLabelRegion \[label the audio sample belongs t, 0 if audio sample is located before the first label]	

_Examples:_
Works for Bach prelude in C but stops during comparison between beats and downbeats tracking algorithms results for Bach prelude in Eb
