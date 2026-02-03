# Designing a modern video codec with Claude

Inspired by:

1) a recent conversation with Tom Dewey about agents
2) the "AI will make everyone a PhD virologist" hype

I thought I'd have a go at finding out whether AI makes everyone a PhD video codec designer. This feels like a similar difficulty goal to 2, but easier to verify, less fraught with guardrails, and *much* less likely to get you put on a list. I started off with the following prompt, complete with a little pep-talk per Tom's advice:

> You are a world-class video codec architect. You have served on advisory boards responsible for the AVC, HEVC and VVC standards, and are deeply familiar with the work that has been done to create "patent free" codecs like AV1 and AV2. Now, you are starting out on your own, and want to create an entirely new codec. You're concerned to avoid the patents that encumber AVC, HEVC and VVC, and want to avoid being seen to "rip off" the design of other existing codecs. You want the codec to achieve VVC- or AV2-like levels of efficiency across a range of bitrates from 1-100Mbps and resolutions between 720p and 4k. You want it to be amenable to "low cost" decoding, using a small (~1mm^2 in 16nm) accelerator, or one or two cores of an Arm Cortex-A76 architecture, along with a mobile-class GPU capable of 50GFLOPs of compute (theoretical) and 2Gtexels/s of texture lookup rate (theoretical). The codec should be a "bit exact" codec, in which a given valid bitstream has a unique valid decoded series of output frames. Can you start by giving me some thoughts on the high-level design of the codec, which we can discuss together?

We went back and forth a little bit, and then I had it spit out a "standards document", which I then fed into Claude Code. In order, we then developed:

- a reference decoder
- a bitstream encoder
- a test harness for the bitstream encoder/decoder pair
- the encoder pixel pipeline
- a complete encoder
- a small test suite

Next we: ground the bugs out of the encoder and decoder; buffed the encoder with some nice-to-have features (full RDO for block partitioning and mode choice, trellis-like squashing of isolated small coefficients, perceptual quality metrics); had a go at tuning the wacky CNN-based loop filter using PyTorch; and did a few rounds of iteration, writing "memos to the architecture team" to flush ambiguity out of the spec.

Things that didn't work perfectly out of the gate and required my input:

- We ended up very dependent on the loop filter to recover losses due to the simplicity of the rest of the design. This may be an interesting approach, but Claude badly underestimated the complexity (9TFOPS vs the 50GFLOP budget), and the cut-down filter feels underpowered.
- The initial proposal was to use texture filtering to implement sub-pel lookups, but this isn't sufficiently specified in the standards, which fights with the "bit exact" goal.
- Claude missed that we needed to perceptually quantise the DCT coefficients

 
