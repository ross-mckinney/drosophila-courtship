
# Introduction

![](_static/ritual.png)

Courtship in *Drosophila melanogaster* consists of an extremely rich series of
stereotyped behaviors, the analysis of which---in conjunction with genetic
and neural manipulations---can provide insights into the sensory stimuli and
brain circuitry that coordinate complex behavioral decisions. Courtship has
traditionally been hand-scored in a binary manner, whereby a male fly is either
courting or not courting a target female. The vast majority of studies which
use courtship as a behavioral output use these simple binary data to calculate
two main parameters: the fraction of time a male spends courting a female
(termed the courtship index) and the time taken for the male to initiate
courtship (called the courtship latency). While these parameters have been
instrumental in advancing diverse areas of research, they are certainly
not capable of capturing the the full complexity and relationship of individual
courtship behaviors to one another.

The software in this repository (*drosophila-courtship*), is designed to help analyze the relationships between individual courtship behaviors displayed by male flies. Particularly, it is useful for tracking males who are courting females that are fixed in place. This allows for the identification of behaviors that are not dependent on female motion. It also allows for the identification of video frames that contain a male engaging in a paritcular behavior towards the female, and the mapping of those frames onto spatial coordinates around the female.
