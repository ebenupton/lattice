# Designing a modern video codec with Claude

Inspired by:

1) a recent conversation with Tom Dewey about agents
2) the "AI will make everyone a PhD virologist" hype

I thought I'd have a go at finding out whether AI makes everyone a PhD video codec designer. This feels like a similar difficulty goal to 2, but easier to verify, less fraught with guardrails, and *much* less likely to get you put on a list. I started off with the following prompt, complete with a little pep-talk per Tom's advice:

> You are a world-class video codec architect. You have served on advisory boards responsible for the AVC, HEVC and VVC standards, and are deeply familiar with the work that has been done to create "patent free" codecs like AV1 and AV2. Now, you are starting out on your own, and want to create an entirely new codec. You're concerned to avoid the patents that encumber AVC, HEVC and VVC, and want to avoid being seen to "rip off" the design of other existing codecs. You want the codec to achieve VVC- or AV2-like levels of efficiency across a range of bitrates from 1-100Mbps and resolutions between 720p and 4k. You want it to be amenable to "low cost" decoding, using a small (~1mm^2 in 16nm) accelerator, or one or two cores of an Arm Cortex-A76 architecture, along with a mobile-class GPU capable of 50GFLOPs of compute (theoretical) and 2Gtexels/s of texture lookup rate (theoretical). The codec should be a "bit exact" codec, in which a given valid bitstream has a unique valid decoded series of output frames. Can you start by giving me some thoughts on the high-level design of the codec, which we can discuss together?

The full transcript can be found here:

https://claude.ai/share/9fdb9d20-4a35-4889-a623-b03fd59a5009

We went back and forth a little bit, and then I had it spit out a "standards document", which I fed into Claude Code. In order, we then developed:

- a reference decoder
- a bitstream encoder
- a test harness for the bitstream encoder/decoder pair
- an encoder pixel pipeline
- a complete encoder
- a small test suite

Next we ground lots of bugs out of the encoder and decoder; buffed the encoder with some nice-to-have features (full RDO for block partitioning and mode choice, trellis-like squashing of isolated small coefficients, perceptual quality metrics); ran the codec off against JPEG and WebP for still images; had a go at tuning the wacky CNN-based loop filter using PyTorch; and did a few rounds of iteration on the spec, writing "memos to the architecture team" to flush out ambiguity.

Things that didn't work perfectly out of the gate and required my input:

- We ended up very dependent on the loop filter to recover losses due to the simplicity of the rest of the design. This may be an interesting approach, but Claude badly underestimated the complexity (9 TFLOPS vs the 50 GFLOPS budget), and the cut-down filter feels underpowered.
- The initial proposal was to use texture filtering to implement sub-pel lookups, but this isn't sufficiently specified in the standards, which fights with the "bit exact" goal.
- Claude missed that we needed to perceptually quantise the DCT coefficients. The initial draft was outperformed on perceptual quality by JPEG for still images despite reporting better PSNR.
- Various minor misunderstandings, such as whether motion compensation is applied to filtered or unfiltered prior frames.

Overall a solid result, with about eight hours of effort (and a lot of tokens) invested. Claude's hyperbolic claims to have developed a VVC-equivalent codec aside, we seem to have ended up with a sort of MPEG-1 on steroids: DCTs and half-pel bilinear filters for motion compensation, but with hierarchical block partitioning, a modern context-adaptive entropy coding scheme, and a loop filter. The reference decoder pixel pipeline is about ~1k lines of quite readable C, plus a few hundred lines of bitstream parser and other support code.

This was my first limited experience with the orchestration model of AI-assisted engineering, and I was quite pleased with how much I could get done in a short period of time. I think it helped that this problem lay at the intersection of three things the tools excel at: strip-mining the literature for concepts; translating between representations (structured natural language and code) while preserving meaning; and beating bugs out of matched pairs of programs. It also helped that I am a (lapsed) video codec engineer; while I'm not intimately familiar with the latest standards, I do know the shape of the problem quite well, and could provide detailed feedback during the requirements and specification phase. Next I might have a go at dropping a real standards document and set of test streams into Claude Code and see how far it can get in producing a conformant implementation.
