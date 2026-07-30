\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{amsmath}

\begin{document}

\thispagestyle{empty}
\vspace*{0.5cm}

\noindent\rule{\textwidth}{2pt}

\vspace{0.45cm}
\begin{center}
  {\LARGE\bfseries Finding Usable Weight Mechanisms with Tiled SVD}
\end{center}
\vspace{0.35cm}

\noindent\rule{\textwidth}{0.8pt}

\vspace{0.6cm}

\begin{center}
\begin{minipage}[t]{0.3\textwidth}
\centering
{\bfseries Ash Manvi}\par
Aquin Labs\par
{\ttfamily ash@aquin.app}
\end{minipage}
\end{center}

\vspace{0.55cm}

\begin{center}
  {\Large\bfseries Abstract}
\end{center}

\begin{quote}
The dominant approach to mechanistic interpretability trains proxy dictionaries
such as sparse autoencoders and labels features from max-activating text. The
best such atlases identify concepts, but their identity lives in the learned
dictionary rather than in the network weights themselves. We propose extracting
\emph{mechanism mounts} directly from linear sites by column-tiled SVD: each
mount is a triple \((v, u, \sigma)\) read as trigger, write, and strength.
Identity is the weight rule. We evaluate mounts with a pre-registered suite
judged on full-write energy lift rather than tile-local lift. On Gemma-2-2B
with WikiText-2 (16{,}384-token subsample), all seven linear maps are scored:
residual writes (\texttt{mlp.down}, \texttt{attn.o}) receive full A/B/C with
steer after post-sublayer RMSNorm and pass \textbf{52/52} site-layers; other
maps receive A/B only (\texttt{mlp.gate}/\texttt{attn.q}/\texttt{attn.k}/effective
\texttt{mlp.up}/\texttt{attn.v} 26/26 each).
Aggregate: \textbf{182/182} GO. We release
library code, the corpus builder, the experiment entrypoint, and unit tests.
\end{quote}

\section{Introduction}

Mechanistic interpretability often asks what a direction in a network
\emph{means}. The standard answer trains a proxy dictionary, typically a
sparse autoencoder, and labels each feature from text that maximizes its
activation. That yields a concept atlas: useful for naming patterns, but with
identity tied to the learned dictionary and its verbal labels rather than to
any particular weight matrix in the model.

This paper builds a different object: a \emph{mechanism atlas}. For a linear
map \(W\) inside the network, we extract \emph{mounts}: rank-1 pieces of \(W\)
that specify when the site writes (a trigger \(v\) on a column tile), where it
writes (a write direction \(u\)), and how strongly (\(\sigma\)). A mount is
used on-distribution when it explains residual write energy above a random
baseline on real forwards. Algebraic SVD identity alone is not enough; that
identity holds almost always by construction.

We ask three questions:
\begin{enumerate}
\item Under a matched mount budget, does tiling input columns before SVD
yield higher full residual write energy lift than whole-matrix SVD, column
sampling, or random directions?
\item Does write coverage saturate with few modes per tile, or keep rising
indefinitely?
\item When does steering along \(u\) move final logits in the direction
predicted by the final unembed of \(u\)?
\end{enumerate}

We do not claim that SVD of weights is a new idea, that mounts carry
human-readable concept names, or that the method replaces sparse autoencoders
for concept discovery. Mounts are identified by layer, site, and mount id.
Reported pass criteria cover all seven linear maps on Gemma-2-2B: residual
writes (\texttt{mlp.down}, \texttt{attn.o}) receive A/B/C; other maps receive
A/B only, with effective-path mounts for \texttt{mlp.up} and \texttt{attn.v}.
Our contribution is the measurement protocol: fair chunking under full-write
energy lift, coverage saturation, and a depth-conditioned causal check against
the final unembedding on residual writes.

\end{document}
