Pinned, unmodified CPS author files from commit
`7ec2be108f4f032f2bb10a6647304076f406ef6b` of
[laymond1/cps-prompt](https://github.com/laymond1/cps-prompt).

| Local file | Original path |
| --- | --- |
| method.py | models/cps_prompt.py |
| model.py | models/cps_prompt_utils/model.py |
| prompt.py | models/prompt_utils/prompt.py |
| vit.py | models/prompt_utils/vit.py |
| cps.py | models/cps_prompt_utils/cps.py |
| cps_vit.py | models/cps_prompt_utils/cps_vit.py |

Byte hashes are enforced by `scripts/cps_author_reference.py`. Original
whitespace and definitions are preserved. The adapter removes top-level imports
and supplies explicit independent dependencies; it uses the source prompt,
attention, sampler, network forward and observe bodies. Offline weight loading,
private RNG scoping, scheduler dispatch and checkpoint counters are documented
framework adaptations. This is a common-protocol reference, not an untouched
author environment or a paper-accuracy reproduction. See LICENSE and NOTICE.md.
