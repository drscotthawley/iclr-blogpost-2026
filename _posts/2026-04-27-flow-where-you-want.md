---
layout: distill
title: "Flow Where You Want"
description: "This tutorial demonstrates how to add inference-time controls to pretrained flow-based generative models. Using an unconditional MNIST flow model, we apply classifier guidance and inpainting by adding velocity corrections during sampling. We also explore PnP-Flow, which satisfies constraints through iterative projection rather than velocity correction."
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous Authors
    url: "https://en.wikipedia.org"
    affiliations:
      name: Anonymous Institution

# must be the exact same name as your blogpost
bibliography: 2026-04-27-flow-where-you-want.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
      - name: The Latent Flow-Matching Setup
      - name: Projecting and Correcting
      - name: Mathematical Details
  - name: Classifier Guidance
    subsections:
      - name: Setup the Models 
      - name: Train a Latent Classifier
      - name: Latents-Only Guidance
  - name: Inpainting
    subsections: 
      - name: Pixel Space Inpainting
      - name: Latent Space Inpainting
      - name: "PnP-Flow: Guidance By Any Other Name"
  - name: Summary

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
  .callout-block {
    width: 85%;
    margin: 25px auto;           
    padding: 15px 30px;          
    background-color: #1f2937;
    color: #f3f4f6;
    border-radius: 8px;    
  }
---




<div style="text-align: center; margin: 0;">
This post is also an executable notebook.<br>
Anonymized Colab Link:
<a href="https://colab.research.google.com/drive/1QkU7NB3eqlPijv1b5GKuC97qdBUzzDVc?usp=sharing" target="_">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="display: inline-block;"/></a>
<br><br>
</div>


# Introduction

In this tutorial, we'll explore inference-time "plugin" methods for flow matching and rectified flow generative models like FLUX or Stable Audio Open Small. Unlike classifier-free guidance (CFG) <d-cite key="cfg"></d-cite>, which requires training the model with your desired conditioning signal, these plugin guidance methods let you add controls at inference time—even for conditions the model never saw during training.

This tutorial assumes familiarity with flow-based generative models, by which we mean "flow matching" <d-cite key="lipman2023flow"></d-cite> and/or "rectified flows" <d-cite key="rectified_flow"></d-cite>.  See the blog posts ["Flow With What You Know"](https://drscotthawley.github.io/blog/posts/FlowModels.html) <d-cite key="hawley2025flowwithwhat"></d-cite> and "A Visual Dive into Conditional Flow Matching" <d-cite key="gagneux2025cfm"></d-cite> for accessible overviews.
The key insight is that flow models generate samples through iterative integration, and at each step we can add small velocity corrections to steer toward specific goals. This works for various objectives: generating specific classes, filling in missing regions, or satisfying other desired constraints.

Our discussion will bring us up to date on guidance methods for latent-space rectified flow models.  While there's an extensive literature on guidance for diffusion models <d-cite key="daras2024survey"></d-cite><d-cite key="ye2024tfg"></d-cite> -- see Sander Dieleman's excellent blog post <d-cite key="dieleman2022guidance"></d-cite> for an overview --- flow matching allows us to cast these in a more accessible and intuitive way. There's some recent work unifying guidance for diffusion and flows <d-cite key="zander_greedy"></d-cite>, but in this tutorial we'll focus on a simplified treatment for flows only.

The paradigm of latent generative models is covered in another superb Dieleman post <d-cite key="dieleman2025latents"></d-cite>, and
combining latent-space models with flow-based guidance gives us powerful, flexible tools for adding flexible controls to efficient generation. 

Let's review the picture for flow-based generative modeling in latent space...



## The Latent Flow-Matching Setup

The following diagrams illustrate the three key concepts: 

**a)** A VAE compresses pixel-space images into compact latent representations. "E" is for encoder and "D" is for decoder:


{% include figure.liquid 
path="assets/img/2026-04-27-flow-where-you-want/vae_flow_diag-VAE.svg"
 class="img-fluid" %}


**b)** The flow model operates in this latent space, transforming noise ("Source", t=0) into structured data ("Target", t=1) through iterative integration. The decoder then converts the final latents back to pixels.

{% include figure.liquid 
path="assets/img/2026-04-27-flow-where-you-want/vae_flow_diag-Generation.drawio.svg"
 class="img-fluid" %}


**c)** While general flows can follow curved trajectories, some of our methods will focus on flows with nearly straight trajectories
which allows for estimating endpoints without many integration steps:

{% include figure.liquid 
path="assets/img/2026-04-27-flow-where-you-want/vae_flow_diag-FlowTypes.drawio.svg"
 class="img-fluid" %}



These (nearly) straight trajectories can be obtained by "ReFlow" distillation of another model (covered in <d-cite key="rectified_flow"></d-cite><d-cite key="hawley2025flowwithwhat"></d-cite> or by insisting during training that the models yield paths agreeing with Optimal Transport such as the "minibatch OT" method of Tong et al <d-cite key="tong2024improving"></d-cite>.  Even if the model's trajectories aren't super-straight, we'll see that the guidance methods we use can be applied fairly generally anyway. 


## Projecting and Correcting

Intuitively, guidance amounts to "steering" during the integration of the flow model in order to end up at a desired end point. Think of paddling a kayak on a river: you look ahead to see where the current is taking you, and if that's not where you want to go, you steer to correct your course. 


The analogy's not quite right: you can't just steer, you are going to have to paddle a little bit.  In other words, you're going to have to provide a bit of a *extra velocity* to correct the where the "current" flow is taking you.


In flow matching, we go from a source data (distribution) at time $$t=0$$ to target
data at $$t=1$$.  Since this tutorial applies to latent space, we'll use the letter $$z$$ for position, such as $$z_t$$ being the position at time $$t$$.

When you're "looking ahead" to estimate where you'll end up, you project linearly along the current velocity $$\vec{v_t}$$ for a duration of the remaining time. Let's call this estimate $$\widehat{z_1}$$, your projected endpoint :

$$ \widehat{z_1} = z_t + (1-t)\vec{v_t}\tag{1} $$

...but perhaps that's not where you want to go. Where you want to go is a distance $$\Delta \widehat{z_1}$$ from $$\widehat{z_1}$$, and to get there you'll have to make a"course correction" $$\Delta \hat{v}$$, as shown in the following diagram:  

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/guidance_vectors.png" class="img-fluid" %}


By similar triangles, $$\Delta \widehat{z_1} = (1-t)\Delta \vec{v}$$, which means the course correction you want is

$$\Delta \vec{v} = { \Delta \widehat{z_1} \over 1-t }\tag{2}$$


<div class="callout-block" markdown="1">
###  Mathematical Details

*Since you're going to see more math once you try to read the scholarly literature on these topics, let's go a bit further into the math...*

So $$\Delta \widehat{z_1}$$ is a measure of the *deviation* from the desired endpoint. Now, in practical application we won't actually use the "distance" $$\Delta \widehat{z_1}$$, but we'll use something that functions *like* a distance, such as a K-L divergence or Mean Squared Error (MSE), which are familiar loss functions from neural network training.

When doing inference, this deviation serves the same function as a "loss" does when training models something we will seek to minimize -- via gradient descent! -- except we'll vary the flow positions $$z$$ instead of the model weights.  More specifically, we'll consider the "likelihood"
 $$p( \widehat{z_1} | y )$$ of getting a $$z_1$$ that matches a given control $$y$$, and we'll seek to maximize that likelihood, or equivalently to *minimize the negative* log-likelihood.

 The expression $$-\nabla_{\widehat{z_1}} \log p( \widehat{z_1} | y )$$ essentially answers the question, "in which direction should I adjust $$\widehat{z_1}$$ so as to make
  $$p( \widehat{z_1} | y )$$ more likely? Just like with gradient descent when training a network,
  this gives us a direction and a magnitude, which we then multiply by a "guidance
  strength" $$\eta$$ (similar to a "learning rate" for gradient descent) to turn it into a step size.
</div>

Applying this gradient-based approach, our expression for $\Delta v$ will involve replacing $\Delta \widehat{z_1}$ in (2) with
  $- \eta \nabla_{\widehat{z_1}} \log p( \widehat{z_1} | y$:

$$ \Delta \vec{v} =  - \eta {1 \over 1-t } \nabla_{z_t} \log p( \widehat{z_1} | y ) \tag{3} $$

where we used the fact that $\nabla_{\widehat{z_1}} = \nabla_{z_t}$ (since $\widehat{z_1} \propto z_t$). The factor of $1/(1-t)$ means small corrections suffice early on, but later times require larger adjustments—though other time scalings are possible, as we'll see.

Now let's apply this to a concrete example.


## Classifier Guidance

If we want our model to generate a member of a particular class, we can use an external classifier to examine the generated samples. The constraint to minimize will be the difference between the desired class and the `argmax` of the classifier output (or some similar relationship that enforces the class compliance).

For our flow model, we'll use [the winning submission](https://github.com/Ocrabit/dl_class_projects/blob/main/dl_experimentation/submissions/marco_submission.py) from the [2025 DLAIE Leaderboard Contest](https://2025-dlaie-leaderboard.streamlit.app/) on unconditional latent flow matching of MNIST digits.  For the classifier, we'll use the [official evaluation classifier](https://github.com/DLAIE/2025-LeaderboardContest/blob/main/evaluate_submission.py) from the same contest.


### Setup the Flow Model and Classifier

Let’s generate and draw some sample images.

<details>
<summary>Show Code</summary>
{% highlight python %}
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# generate some samples
n_samples = 10
x1 = sub.generate_samples(n_samples=n_samples)
x1.shape


def show_grid(x1, title=None):
    if len(x1.shape) == 3: x1 = x1.unsqueeze(1)  # add channels dim
    grid = make_grid(x1, nrow=10, padding=2, normalize=False)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
    plt.axis('off')
    if title: plt.title(title)
    plt.tight_layout()
    plt.show()

show_grid(x1, "Sample generated images")
{% endhighlight %}

</details>

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/gen_image_row_1.png" class="img-fluid" %}

Now we'll setup the (pretrained) classifier we’ll use for the guidance:
Let’s make a plot showing the classifier’s output probabilities (aka likelihoods) across all classes, for all 10 samples. The samples will be the rows, and the class-likelihoods outputs from the classifier will be the columns, where brightness is correlated with likelihood.

<details><summary>Show Code</summary>
{% highlight python %}
def show_probs(probs, x=None):
    """show probs as colormap intensities via imshow.
    have each row be a sample and each column be a class probability"""
    ncols = 1 if x is None else 2
    fig, axs = plt.subplots(1, ncols, figsize=(8, 4))
    if ncols == 1: axs = [axs]

    if x is not None:  # show a little version of the x image for each row
        axs[0].imshow(make_grid(x.unsqueeze(1).cpu(), nrow=1, padding=2, normalize=False).permute(1, 2, 0).cpu(), cmap='gray')
        axs[0].axis('off')

    # show probabilities as an intensity map
    im = axs[ncols-1].imshow(probs.cpu(), cmap='gray')
    axs[ncols-1].set_xlabel("Class")
    axs[ncols-1].set_ylabel("Sample #")
    plt.colorbar(im, ax=axs[ncols-1])
    plt.tight_layout()

show_probs(probs, x=x1)
{% endhighlight %}
</details>


{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/show_probs_1.png" class="img-fluid" %}


From the “random” distribution of generated digits, we see that this is an unconditional generative model: there’s nothing determining the classes of the outputs – until we add guidance, below! In a short while, we’ll reproduce that diagram, but we’ll use guidance to get one class per sample, in order, along the diagonal.

To do that, we're going to have to "break open" the `generate_samples` routine and even the `integrate_path` routine to allow us to *add a correction* to the velocity $v_t$ generated by the flow model at time $t$.  That correction 
$\Delta v$  will be based on the classifier's output using the *projected estimate* $\widehat{x_1}$ of the final data, which we'll obtain via *linear extrapolation*.  

In our latent space model, we flow with latents $z$ which must be *decoded* using the VAE's decoder $D$ :

$$\widehat{z_1} = z_t + (1-t) v_t$$

$$\widehat{x_1} = D(\widehat{z_1})$$


The correction $\Delta v$  will generated from a constraint which in this case is just like regular "classifier loss" function in a supervised learning problem. The desired class label is the "target" and the classifier output of the projected estimate is the "prediction".

Our code will follow this general layout: 

{% highlight python %}

loss_fn = torch.nn.CrossEntropyLoss()

v_t = flow_model(z_t, t)
z1_hat = z_t + (1-t)*v_t               # projected destination
x1_hat = sub.vae.decoder(z1_hat)       # decode it to pixel space
probs = classify(classifier, x1_hat)   # classifer operates in pixel space
loss = loss_fn(probs, target)          # "supervised learning"
delta_v = magic_function(loss,...???)  # <--- here's the part we need to work out
v_t = v_t + delta_v * guidance_strength  # we can set the strength of the correction
{% endhighlight %}


To convert that `loss` into a "velocity," we can take its gradient. When training a model, one typically takes the gradient with respect to the model weights. For inference-time guidance,
however, we will take the gradient with respect to the flow coordinates $z$  in the latent space, thereby generating a vector in the latent space.

PyTorch lets us compute the gradient with respect to anything (in the autograd graph). We just need to tell it what we want. And we need to be careful to make sure that the VAE and flow models stay frozen, so the only thing that's allowed to change are the latents $z$.

The cleanest way to pull this off, code-wise, is to create a function called `compute_v()` which for starters will just call the flow model, but then we'll add to it with guidance info:

{% highlight python %}
# wherever we used to just call flow_model(), we'll now call compute_v() instead
@torch.no_grad()
def compute_v(flow_model, z, t, guidance_dict=None, **kwargs):
    v_t = flow_model(z, t)
    if guidance_dict is not None:
        v_t += compute_dv(v_t, z, t, guidance_dict, **kwargs)
    return v_t

@torch.enable_grad()  # <-- later, this will be a key for getting guidance
def compute_dv(v_t, z, t, g:dict, **kwargs):
    "placeholder for now, will add guidance math later"
    return torch.zeros_like(v_t).detach() # no correction yet; no gradients returned
{% endhighlight %}

We'll to use some typical "boilerplate" flow integration code, except we'll add "`**kwargs`" everywhere so we can pass controls "all the way in" to the `compute_dv()` guidance routine, and pair `flow_model()` as an arg to `compute_v()` via `functools.partial`.


<details><summary>Show Flow Integration Code</summary>
{% highlight python %}

from functools import partial  # use partial to package flow_model with compute_v

@torch.no_grad()
def rk4_step(f, y, t, dt, **kwargs):  # regular rk4, + kwargs passthrough
    # f: callable (y, t) -> dy/dt
    k1 = f(y, t, **kwargs)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = f(y + dt * k3, t + dt, **kwargs)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

@torch.no_grad()
def warp_time(t, dt=None, s=.5):
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s < 0 or s > 1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4 * (1 - s) * t ** 3 + 6 * (s - 1) * t ** 2 + (3 - 2 * s) * t
    if dt:  # warped time-step requested; use derivative
        return tw, dt * 12 * (1 - s) * t ** 2 + 12 * (s - 1) * t + (3 - 2 * s)
    return tw


@torch.no_grad()
def integrate_path(model, initial_points, step_fn=rk4_step, n_steps=100, warp_fn=None, latent_2d=False, prog_bar=True, t0=0, **kwargs):
    p = next(model.parameters())
    device, model_dtype = p.device, p.dtype
    current_points = initial_points.to(device=device, dtype=model_dtype).clone()
    model.eval()
    ts = torch.linspace(t0, 1, n_steps, device=device, dtype=model_dtype)
    if warp_fn: ts = warp_fn(ts)
    if latent_2d: t_batch = torch.empty((current_points.shape[0], 1), device=device, dtype=model_dtype)
    vel_model = partial(compute_v, model)  # here's the secret sauce
    iterator = range(len(ts) - 1)
    if prog_bar: iterator = tqdm(iterator, desc="Integrating Path")
    for i in iterator:
        t, dt = ts[i], ts[i + 1] - ts[i]
        if latent_2d: t = t_batch.fill_(t.item())
        current_points = step_fn(vel_model, current_points, t, dt, **kwargs)
    return current_points

def generate_samples(sub, n_samples: int, n_steps=15, z0=None, t0=0, **kwargs) -> torch.Tensor:
    z0 = torch.randn([n_samples, sub.latent_dim]).to(sub.device) if z0 is None else z0
    z1 = integrate_path(sub.flow_model, z0, n_steps=n_steps, step_fn=rk4_step, t0=t0, **kwargs)
    gen_xhat = F.sigmoid(sub.decode(z1).view(-1, 28, 28))
    return gen_xhat
{% endhighlight %}

</details>



Now that we know that works, let's upgrade `compute_dv()` to include the guidance correction.  We'll use the `torch.autograd.grad()` function to compute the gradient of the loss.  
First we have the `guidance_dict` that we'll use to pass through our intentions through the various layers of routines to get to `compute_dv()`:  

{% highlight python %}
guidance_dict = \
    {'classifier': classifier,     # the classifier model to use
    'decode': sub.decode,         # how to decode to pixel space for classifier
    'loss_fn': torch.nn.CrossEntropyLoss(reduction='none'), # don't sum over batch dim
    'target': torch.arange(10).to(device),  # desired class outcomes
    'strength': 5.0,              # "guidance strength", you may vary this
    't_min': 0.01, 't_max': 0.99, # t range to apply guidance, may vary these
    }
{% endhighlight %}



Next we have the fully-equipped `compute_dv()`. This code is overly-commented to make it easy to follow each step.  (We replaced `guidance_dict` with `g` locally for brevity.)  No other changes to any preceding code are necessary. We'll be ready to do guided inference after this definition!


{% highlight python %}
@torch.enable_grad()  # <-- Needed to compute gradients if calling code has @torch.no_grad()
def compute_dv(v_t, z, t, g:dict, eps=1e-6, debug=False):
    "Compute the guidance correction to the flow velocity"
    if t < g['t_min'] or t > g['t_max']: return torch.zeros_like(v_t).detach()
    z.requires_grad_(True)                   # need to enable gradient tracking for z
    z1_hat = z + (1 - t) * v_t               # linear projection to estimated endpoint

    # Decoding to pixel space (if decoder provided)
    x1_hat = z1_hat if g['decode'] is None else F.sigmoid(g['decode'](z1_hat)).view(-1, 28, 28)

    logits, probs = classify(g['classifier'], x1_hat)          # run classifier
    loss = g['loss_fn'](logits, g['target'][:len(logits)])     # loss <-> "negative log likelihood"

    # Compute grad wrt z. "grad_outputs=": don't sum over over batch, keep unique to each datum
    grad_z = torch.autograd.grad(loss, z, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
    dv = -grad_z / (1 - t + eps)   # - minimizes, (1-t) makes it velocity, eps helps stability

    z.requires_grad_(False)        # cleanup (z is a tensor so local changes could propagate)
    return g['strength'] * dv.detach()  # detach so no gradients returned
{% endhighlight %}

Let's now generate using classifier guidance on the flow model, and visualize the results:

{% highlight python %}
torch.manual_seed(0) # for reproducibility as we change other things
with torch.no_grad():
   x1 = generate_samples(sub, n_samples=10, guidance_dict=guidance_dict, debug=False)
   logits, probs = classify(classifier, x1)
show_probs(probs, x=x1)
{% endhighlight %}


{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/show_probs_2.png" class="img-fluid" %}


Excellent! That was our desired goal: consecutive classes along the diagonal.

To get a better survey of the guidance capabilities, let's make a 10x10 grid of outputs with classes along each column:

{% highlight python %}
target = torch.arange(10).repeat(10).to(device) #  [0,1,2,..9, 0,1,2,..9, ...]
guidance_dict['target'] = target
torch.manual_seed(42)         # (optional) for reproducibility
x1 = generate_samples(sub, n_samples=len(target), guidance_dict=guidance_dict)
show_grid(x1, "Guided samples")
{% endhighlight %}


{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/class_guidance_grid.png" class="img-fluid" %}


That worked fine, and on a GPU it's pretty fast, but on systems with only a CPU, it's *painfully* slow. So instead, let's...



## Train a Latent Classifier
We'll train a model `z_classifier` that looks only in latent space, so we can use it as a guidance signal. This can be a very simple model consisting of a few `Linear` layers:


{% highlight python %}
class LatentClassNet(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, n_classes=10):
        super().__init__()
        dims = [latent_dim, hidden_dim, hidden_dim, latent_dim, n_classes]
        self.layers = nn.ModuleList([nn.Linear(in_d, out_d) for in_d, out_d in zip(dims[:-1], dims[1:])])   
    def forward(self, z):
        for layer in self.layers: z = F.leaky_relu(layer(z))
        return z

z_classifier = LatentClassNet().to(device)
{% endhighlight %}


This classifier will operate on *latent* encodings of the MNIST dataset. So let's save the encoded latents to disk and load them into memory. These will be our training and test data. 

<details><summary>Show Code</summary>
{% highlight python %}
#| code-fold: true

# You can probably skip this code block. It just runs MNIST
# through the VAE's encoder and saves it to disk. 

from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from glob import glob 
import math 

@torch.no_grad()
def encode_dataset(vae, dataset, batch_size=512, chunk_size=10000, tag="train"):
    """Encode dataset into VAE latents (z = mu), saving progress in temp chunk files.
    We use temp chunks in case execution gets interrupted, we can try again & resume. 
    """
    device = next(vae.parameters()).device
    total_chunks = math.ceil(len(dataset) / chunk_size) 
    # check for existing chunk files 
    basename = f"tmp_chunk_{tag}"
    chunk_files = glob(basename+'*.pt') 
    existing_chunks = len(chunk_files)
    print(f"{tag}: Found {existing_chunks} of {total_chunks} expected chunks. Generating remaining...")
    for c in range(existing_chunks, total_chunks): 
        print(f"chunk {c+1}/{total_chunks}:",end="")
        indices = list(range(c*chunk_size, (c+1)*chunk_size ))
        data_subset = Subset(dataset, indices)        
        loader = DataLoader(data_subset, batch_size=batch_size, shuffle=False)
        all_latents, all_labels = [], []
        must_flatten = None
        with torch.no_grad():
            for data, labels in loader:
                print("=",end="") # dumb progress bar 
                x = data.to(device)
                # next bit is so it should work with linear layers or conv
                if must_flatten is None or must_flatten==False:
                    try: z, must_flatten = vae.encoder(x), False
                    except RuntimeError:
                        z = vae.encoder(x.view(x.size(0), -1))
                        must_flatten = True
                else: z = vae.encoder(x.view(x.size(0), -1))
                mu, logvar = z
                all_latents.append(mu.cpu())
                all_labels.append(labels)
        chunk_latents, chunk_labels = torch.cat(all_latents), torch.cat(all_labels)
        tmp_filename = f"{basename}_{c+1}.pt"
        print(f"|  Saving chunk to {tmp_filename}")
        torch.save({ 'latents':chunk_latents, 'labels':chunk_labels }, tmp_filename)
    
    # Assemble all the chunks from files and return tensors.
    print("Assembling", basename+'*.pt ...')
    all_latents, all_labels = [], []
    for chunk_file in glob(basename+'*.pt'): 
        chunk_dict = torch.load(chunk_file, weights_only=True)
        all_latents.append(chunk_dict['latents'])
        all_labels.append(chunk_dict['labels']) 
    latents, labels = torch.cat(all_latents), torch.cat(all_labels)
    #for f in glob(tmp_file_base+'*.pt'): os.remove(f)   # clean up 
    return latents, labels 

def encode_mnist(vae, filename=None, batch_size=512):
    print("Acquiring train & test MNIST image datasets...")
    train_ds = MNIST(root='./data', train=True,  download=True, transform=ToTensor())
    test_ds  = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    
    print(f"\nEncoding dataset to latents...")
    train_latents, train_labels = encode_dataset(vae, train_ds, batch_size=batch_size, tag='train')
    test_latents, test_labels = encode_dataset(vae, test_ds, batch_size=batch_size, tag='test')
    for f in glob('tmp_chunk_t*_c*.pt'): os.remove(f)  # clean up

    if filename is not None:
        print(f"Saving to {filename} ...")
        torch.save({ 'train_z': train_latents,     'test_z': test_latents,
                     'train_labels': train_labels, 'test_labels': test_labels }, filename)
    return train_latents, train_labels


# Encode the dataset
latent_data_filename = 'mnist_latents.pt'
if not os.path.exists(latent_data_filename):
    train_latents, train_labels = encode_mnist(sub.vae, filename=latent_data_filename)


def load_data(filename):
    if 'MyDrive' in filename:
        from google.colab import drive
        drive.mount('/content/drive')
    data_dict = torch.load(filename, weights_only=True)
    return data_dict

data_dict = load_data(latent_data_filename)
train_z, test_z = data_dict['train_z'], data_dict['test_z']
train_z.shape, test_z.shape

# Create datasets from the latent tensors
train_latent_ds = TensorDataset(train_z, data_dict['train_labels'][:train_z.shape[0]])
test_latent_ds = TensorDataset(test_z, data_dict['test_labels'])

batch_size = 512
train_latent_dl = DataLoader(train_latent_ds, batch_size=batch_size, shuffle=True)
test_latent_dl = DataLoader(test_latent_ds, batch_size=batch_size, shuffle=False)

print(f"Train batches: {len(train_latent_dl)}, Test batches: {len(test_latent_dl)}")
# print single latent size
print(f"Latent size: {train_latent_ds[0][0].shape}")
{% endhighlight %}  
</details>




Then we'll run the training loop...

<details>
<summary>Training Loop Code and Execution</summary>
{% highlight python %}
# Latent classifier training loop
optimizer  = torch.optim.Adam(z_classifier.parameters(), lr=1e-3, weight_decay=1e-5)
criterion  = nn.CrossEntropyLoss()
epochs = 8
for epoch in range(epochs):
    z_classifier.train()
    pbar = tqdm(train_latent_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for latents, labels in pbar:
        optimizer.zero_grad()
        logits = z_classifier(latents.to(device))
        loss   = criterion(logits, labels.to(device))
        pbar.set_postfix({'train_loss': f"{loss.item():.4f}"})
        loss.backward()
        optimizer.step()
    
    # Validation
    z_classifier.eval()
    val_latents, val_labels = next(iter(test_latent_dl))
    val_logits = z_classifier(val_latents.to(device))
    val_loss   = criterion(val_logits, val_labels.to(device))
    val_acc    = (val_logits.cpu().argmax(dim=1) == val_labels).float().mean()
    print(f"Epoch {epoch+1}/{epochs}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, val_acc={val_acc.item():.4f}")
{% endhighlight %}


```
Epoch 1/8: train_loss=0.9673, val_loss=1.1017, val_acc=0.6211
Epoch 2/8: train_loss=0.1901, val_loss=0.1919, val_acc=0.9375
Epoch 3/8: train_loss=0.0512, val_loss=0.1205, val_acc=0.9570
Epoch 4/8: train_loss=0.1193, val_loss=0.1022, val_acc=0.9668
Epoch 5/8: train_loss=0.0810, val_loss=0.0948, val_acc=0.9648
Epoch 6/8: train_loss=0.1569, val_loss=0.0815, val_acc=0.9707
Epoch 7/8: train_loss=0.0504, val_loss=0.0841, val_acc=0.9629
Epoch 8/8: train_loss=0.0408, val_loss=0.0792, val_acc=0.9746
```
</details>


Let's test our newly-trained latent classifier, to make sure it works before trying to use it for guidance.  We'll pull up data samples with known ground-truth "target" labels, and compare these to the predictions from the classifier.  If the targets and predictions match up, we're good to go:

{% highlight python %}

z, L = test_latent_ds[20:30]
z = z.to(device) 
show_grid(F.sigmoid(sub.decode(z)).view(-1,28,28))
with torch.no_grad():
    pred_class = classify(z_classifier, z, use_argmax=True)
print("Target labels:   ",L)
print("Predicted labels:", pred_class.cpu())
{% endhighlight %}

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/gen_image_row_2.png" class="img-fluid" %}

```
Target labels:    tensor([9, 6, 6, 5, 4, 0, 7, 4, 0, 1])
Predicted labels: tensor([9, 6, 6, 5, 4, 0, 7, 4, 0, 1])
```
Good! They match up.  Let's move on...


### Latents-Only Guidance

Now that we have a trained classifier that operates in latent space, we can run basically the same code as before, only it will execute much faster...


<details><summary>Show Code</summary>
{% highlight python %}
guidance_dict ={'classifier': z_classifier,
                'decode': None,    # no decoding, latent space only
                'loss_fn': torch.nn.CrossEntropyLoss(reduction='none'), # don't sum across batch dim
                'target': torch.arange(10).repeat(10).to(device),
                'strength': 5.0,   # "guidance strength"
                't_min': 0.01,  't_max': 0.99, }

torch.manual_seed(42) # remove for new samples each time
x1 = generate_samples(sub, n_samples=len(guidance_dict['target']), guidance_dict=guidance_dict)
show_grid(x1, "Latent-Only Guidance")
{% endhighlight %}
</details>

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/latent_guidance_grid.png" class="img-fluid" %}

That executes very quickly, even on a CPU, and the results are just as good as before.
Since we no longer have to propagate gradients through the much larger VAE decoder model and pixel-space classifer, we can get answers a lot faster via our small latents-only classifier.

Next, we'll explore another application of guidance, for which our guidance signal doesn't depend on a separate trained (classifier) model at all: inpainting.



## Inpainting

When inpainting, we have some "mask" inside which some of the data have been removed, and we want to use the model to fill in the missing part in a way that matches with the surrounding pixels. Let's take a look at an example from MNIST, where we show an original image, the mask and the masked-out image:


<details><summary>Show Code</summary>
{% highlight python %}
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

test_ds  = MNIST(root='./data', train=False, download=True, transform=ToTensor())
x = test_ds[7][0]
H, W = x.shape[-2:]
M = torch.ones([H,W], dtype=x.dtype, device=x.device)   # 1 = keep pixels
M[H//3:2*H//3, W//3:2*W//3] = 0                         # 0 = mask out
x_masked = M*x
show_grid( torch.cat([x, M.unsqueeze(0), x_masked],dim=0),"      Original      |      Mask      |  Masked Image" )
{% endhighlight %}
</details>
{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/inp_mask_example.png" class="img-fluid" %}




Another example would be a picture of a face where you've blocked out the nose and you want the model to fill in a nose.  Some of the "filling in" is obtained nearly "for free" because the model has only been exposed to data that satisfies the manifold or probability distribution of the training data -- e.g. If it was trained on faces, then it only ever saw faces with noses and hence can only generate faces with noses -- but the real trick is to do it "well" and have the inpainted region match up nicely with the surrounding pixels.


There's a wealth of information on guidance as it was originally applied to diffusion models. We recommend Sander Dieleman's blog post, ["Guidance: a cheat code for diffusion models"](https://sander.ai/2022/05/26/guidance.html), for an extremely informative survey.  Yet because of the stochastic/random nature of the diffusion path, there are several "complicating" aspects of diffusion guidance that we're going to gloss over in this tutorial because in the case of deterministic, smooth flow-model trajectories, things become a lot more intuitive.



We'll follow a method by Pokle et al <d-cite key="pokle2024trainingfree"></d-cite>, a method that applies to general linear inverse problems of which inpainting is a particular case, and we'll simplify their method to adapt it for *just inpainting.* For a more rigorous treatment connecting flow-based inverse problem solving to posterior sampling, see also FlowDPS <d-cite key="flowdps"></d-cite>, which extends diffusion inverse solvers into the flow framework.



The method involves generating an *entire* new image $$x_1$$ that everywhere *outside the mask matches up* with the pixels in user-supplied (masked) image $$y$$.  So the constraint will be, given a 2D mask $$M$$ (where $$M$$=1 means there's an original pixel there, and $$M$$=0 is the masked-out region), to require that our estimate image $$\widehat{x_1}$$ (i.e. the decoded image version of the estimated latents $$\widehat{z_1}$$   ) satisfies  $$M*\widehat{x_1} = M* y$$ <d-footnote>where "$*$" denotes the elementwise or Hadamard product</d-footnote>, or in a "residual form", we'll just compute the Mean Squared Error (MSE) of $$M*(\widehat{x_1}-y)$$:

$$ {\rm Constraint:} = M^2 * (\widehat{x_1}-y)^2 $$

(and if we want, we can use the fact that $$M$$ being a binary mask means $$M^2 = M$$).

If we want to do latent-only inpainting (which will be the fastest), then the same constraint applies just with the simplification $$\widehat{x_1} = \widehat{z_1}$$


The authors of the paper recommend only doing guidance from t equals 0.2 onward because prior to that, it's hard to make any meaningful estimate.. In fact, they don't even integrate before $$t = 0.2$$. They just interpolate between the source and the target data to get their starting point at  $$t = 0.2$$.

To use our constraint in the guidance equation (3) for computing $$\Delta v\,$$, we'll need to turn our constraint into a likelihood by raising it to an exponential power -- so we get a Gaussian! But the guidance equation includes a logarithm that immediately *undoes* our exponentiation:

$$\Delta v = - {\eta \over 1-t} \nabla_{z_t}\   
{\color{red}{\text{l̸o̸g̸}} \, \color{red}{\text{e̸x̸p̸}}}
\left( M^2 * (\widehat{x_1}-y)^2 \right).$$

The gradient part is
 $$ \nabla_{z_t} M^2 *(\widehat{x_1}-y)^2 = 2M^2*(\widehat{x_1}-y) {\partial \widehat{x_1} \over \partial z_t } $$

If we're inpainting in latent space and not using the decoder for the constraint, then $${\partial \widehat{x_1} / \partial z_t } = 1$$.  Otherwise that term will require evaluation via PyTorch's `autograd` (=slow).



Our earlier time scaling was $$\gamma_t = 1/(1-t)$$; turns out that doesn't work very well in practice when it comes to inpainting. Instead, we'll use a different time scaling that delivers good (albeit not perfect) results:  $$\gamma_t = (1-t)/t$$.  

Thus our full equation for the velocity correction will be:

$$ \Delta \vec{v} = -\eta\, \gamma_t\, M^2 *(\widehat{x_1} - y){\partial\widehat{x_1}\over\partial{z_t}}, \ \ \ \ \ \ \ \ \ \gamma_t = {1-t\over t}$$

where we absorbed the factor of 2 into $$\eta$$, and the last partial derivative term can be one if we do latent-only inpainting.

Let's implement this in code, using two different versions of the gradient calculation, depending on whether we can do it all in latent space or if we need to propagate gradients through the decoder:


<details><summary>Show Code</summary>
{% highlight python %}
@torch.no_grad()  # gradients computed analytically!
def ip_latents_grad(v_t, z, t, g:dict, eps=1e-6, **kwargs):
    "gradients for latent-only inpainting, fast"
    z1_hat = z + (1-t)*v_t
    return g['M_sq'] * (z1_hat - g['y'])  #  x1_hat = z1_hat, dz1_hat/dz_t=1

@torch.enable_grad()
def ip_pixels_grad(v_t, z, t, g:dict, eps=1e-6, **kwargs):
    "gradients for pixel-space inpainting. need to use decoder & track via autograd, = slow"
    z.requires_grad_(True)
    z1_hat = z + (1-t)*v_t
    x1_hat = F.sigmoid(g['decode'](z1_hat)).view(-1,1,28,28) # Hard-coded for 28x28; adjust for other datasets
    grad_x = g['M_sq'] * (x1_hat - g['y'])
    grad_z = torch.autograd.grad(x1_hat, z, grad_outputs=grad_x,retain_graph=False)[0] # mults grad_x by dx1_hat/dz1_hat
    z.requires_grad_(False)
    return grad_z.detach()  # don't send gradients onward

def t_timescale(t, timescale='mine', **kwargs):
    "our choice for adaptive time scale"
    if timescale =='simple': return 1/(1-t)                          # our earlier scale; doesn't work
    elif timescale=='pokle': return (1-t)**2 / ((1-t)**2 + t**2)     # from pokle et al; can't get it to work
    elif timescale=='constant': return 4  # or any constant. The 4 is from Pokle et al
    else: return (1-t)/t  # This works pretty well! strong guidance at start -> zero at end

def compute_dv_inpainting(v_t, z, t, g:dict, **kwargs):
    "wrapper to call appropriate gradient-computation routine"
    if t < g['t_min'] or t > g['t_max']: return torch.zeros_like(v_t)
    grad_fn = ip_latents_grad if g['decode'] is None else ip_pixels_grad
    grad = grad_fn(v_t, z, t, g, **kwargs)
    dv = -g['strength'] * t_timescale(t, **kwargs) * grad
    return dv.detach()
{% endhighlight %}
</details>


### Do the Inpainting

<details><summary>Show Code</summary>
{% highlight python %}
y = torch.stack([test_ds[i][0] for i in range(50)])
print(y.shape)
y = M*y
show_grid(y.squeeze(), "Masked Images")
{% endhighlight %}
</details>

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/inp_masked_images.png" class="img-fluid" %}


And now we run the inpainting code...

<details><summary>Show Code</summary>
{% highlight python %}

compute_dv = compute_dv_inpainting  # register our new guidance routine

inpainting_dict ={'decode': sub.decode,         # how to decode to pixel space for classifier
                'M_sq': (M**2).to(device),
                'y': y.to(device),
                'strength': 1.0,              # "guidance strength", you may vary this
                't_min': 0.2, 't_max': 0.999} # t range to apply guidance, may vary these

with torch.no_grad():
    torch.manual_seed(0) # for reproducibility as we change other things
    t0 = 0.2             # starting time as per Pokle et al
    z0 = torch.randn([len(y), sub.latent_dim]).to(sub.device)
    zy = sub.encode(y.to(device))   # encoded version of masked image
    z0 = z0 * (1-t0) + zy * t0      # interpolation init
    inpainting_dict['t_min'] = t0
    x1 = generate_samples(sub, n_samples=len(y), t0=t0, z0=z0, guidance_dict=inpainting_dict, warp_fn=None, debug=False)

show_grid(x1, "Inpainted Images") 
{% endhighlight %}
</details>
{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/inp_inpainted_images.png" class="img-fluid" %}


We see that the generated images generally look great, although in some cases, the in-painting code has changed pixels even where the mask is 1. We can disallow this by just resetting those values to the pixels in $y$.

Turning up the guidance strength would also enforce our constraint better, but turning up too high causes the whole thing to diverge and we get garbage out.

In order to experiment with other methods more easily, we should do inpainting only in latent space, and for that we will need a model that supports spatial latents....

### Latent-Only Inpainting


To do inpainting in latent space, we'll need to switch models to one where the latents preserve the spatial structure of the original images. 

<details>
<summary>Get the spatial-latents model</summary>
{% highlight python %}
# Get Spatial VAE & FLow DiT Model
!wget -q --no-clobber=off https://raw.githubusercontent.com/dlaieburner/2025-leaderboard/refs/heads/main/sample_submission_dit.py

try:
    del SubmissionInterface # remove Marco's from earlier; make it reload
except NameError:
    pass  # nevermind

from sample_submission_dit import SubmissionInterface

sub = SubmissionInterface().to(device)
{% endhighlight %}
</details>

Let's take a look at the images and their spatial-latent representations:

<details><summary>Show Code</summary>
{% highlight python %}
#| code-fold: true

# viz images and spatial latents
from torchvision.datasets import MNIST
test_ds  = MNIST(root='./data', train=False, download=True, transform=ToTensor())

x = torch.stack([test_ds[i][0] for i in range(6)])
if len(x.shape) < 4: x1 = x.unsqueeze(1)
show_grid(x, "Images")
z1 = sub.encode(x)
show_grid((z1-z1.min())/(z1.max()-z1.min()), "Latents")
{% endhighlight %}
</details>

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/spatial_latents_viz1.png" class="img-fluid" %}
{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/spatial_latents_viz2.png" class="img-fluid" %}

Now we can run latent-only inpainting using the same code as before, only this time with the spatial-latent model and with `decode=None` in the guidance dictionary:

<details><summary>Show Code</summary>
{% highlight python %}
def prepare_latent_mask(image, mask, encoder):
    z_y = encoder(image.to(device))
    M_z = F.interpolate(mask[None, None], size=z_y.shape[-2:], mode='bilinear', align_corners=False).to(device)
    M_z = (M_z > 0.9).float()    # binarize it
    return z_y * M_z, M_z**2

@torch.no_grad()
def latents_only_inpaint(sub, inpainting_dict, n_samples, n_steps=20, t0=0.2, warp_fn=None):
    z_y, M_sq = prepare_latent_mask(y, M, sub.encode)   
    inpainting_dict.update({'M_sq': M_sq, 'y': z_y, 't_min': t0})
    z0 = torch.randn_like(z_y) * (1-t0) + z_y * t0         # Initialize via interpolation
    return generate_samples(sub, n_samples, t0=t0, z0=z0,  guidance_dict=inpainting_dict, warp_fn=warp_fn)

inpainting_dict ={'decode': None,            # now we're latents-only
                'strength': 1.0,             # "guidance strength", you may vary this
                't_min': 0.2, 't_max': 0.999} # t range to apply guidance, may vary these

show_grid(y.squeeze(), "pixel y's")
x1 = latents_only_inpaint(sub, inpainting_dict, n_samples=len(y), n_steps=10)
print()
show_grid(x1, "inpainted images")
{% endhighlight %}
</details>

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/latents_only_pixel_ys.png" class="img-fluid" %}
{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/latents_only_inpainted_images.png" class="img-fluid" %}



As before, the latent-only execution is very fast, and the results are quite good.
There are some limitations to what's being produced. The problem is due to the low resolution of our latents. Inpainting algorithms typically assume higher resolution in order to work accurately. But this lesson was designed to run quickly on CPUs and thus there is a trade-off.

Now, if we want to enforce the original pixels where the mask is one, we can do that at every stage of the integration process. We just need to modify the integration process to overwrite $z$ wherever `Mz` is 1.

Next up is a different method that can achieve similar results, albeit differently...




# PnP-Flow: Guidance By Another Name

The term 'guidance' typically refers to velocity modifications, but PnP-Flow by Martin et al <d-cite key="pnp_flow"></d-cite> achieves similar results by adjusting latent positions $z$ directly.<d-footnote>The paper by Pokle et al we cited earlier also included a related method, however the PnP-Flow method isn't restricted to linear problems. Plus, Anne Gagneux provided [code](https://github.com/annegnx/PnP-Flow) for PnP-Flow!  Gagneux's repo even provides code for the position-only (non-velocity) algorithm from Pokle et al aka "OT-ODE".</d-footnote>$$^,$$<d-footnote>Differences between our variables and those in the PnP-Flow paper: For us, $z$ are integrated flow latent variables between $z_0$ (source) and $z_1$ (target), whereas $x$ are the pixel-space representations via our VAE's decoder $D$ such that $D(z)=x$. In PnP-Flow, $x$ is the integrated flow  variable, $z$ is used only for their interpolation/overwrite step, and $D$ is the "denoiser" aka their flow model.</d-footnote>

PnP-Flow assumes straight-line trajectories, making the forward projection trivial: $\widehat{z_1}$
is reached by simple linear extrapolation.  Instead of incrementally moving $z$ from $t=0$ to $t=1$, PnP-Flow projects forward to $\widehat{z_1}$ and iterates on that estimate through a series of correction and projection steps. The first step applies our gradient correction:

$${\rm Step\ 1.}\ \ \ \ \ \  \ \ \ \ \ \ z_1^* := \widehat{z_1} - \eta\,\gamma_t \nabla F(\widehat{z_1},y)$$

where $z_1^*$ (my notation) is our goal i.e. the endpoint of our projected course correction, and $F(\widehat{z_1},y)$ is our (log-exp probability) constraint.  For the time scaling, the PnP-Flow authors recommend $\gamma_t = (1-t)^\alpha$
with $\alpha \in [0,1]$ is a hyperparameter chosen according to the task -- e.g., they use $\alpha$'s as large as 0.8 for denoising tasks, 0.5 for box inpainting, and 0.01 for random inpainting. This choice of $\gamma_t$ is a bit different from our earlier one of $(1-t)/t$. Both go to zero as $t \rightarrow 1$, but approach it differently and have different asymptotics as $t\rightarrow 0$.

In the graph below, we show our earlier choice of $(1 - t)/t$ in green and $(1 - t)^\alpha$ in purple for various choices of $\alpha$:
 

  <center>
  <a href="https://www.desmos.com/calculator/bcp2wiyyid">
  <iframe src="https://www.desmos.com/calculator/bcp2wiyyid?embed" frameborder='0' scrolling='no' height="300px" width="200px"></iframe>
<br>Interactive Desmos Graph Link</a><br><br></center>




...where for "box inpainting" as we did above, they use $\alpha$=0.5.

But PnP-Flow doesn't stop there!  Two other key steps remain. We then project backward to *overwrite* $z_t$ with a corrected value:

$${\rm Step\ 2.}\ \ \ \ \ \  \ \ \ \  \ \ z_t := (1-t)\,z_0 + t\, z_1^* $$

We then compute a new projected estimate, same as we have before:

$${\rm Step\  3.}\ \ \ \ \ \   \widehat{z_1} := z_t + (1-t)\,v_t(z,t)$$

....and loop over Steps 1 to 3 for each value of $t$ in our set of (discrete) integration steps, i.e. after Step 3, we let $t := t+\Delta\,t$ and go back to Step 1. Our final value of $\widehat{z_1}$ will be the output.

This image from the PnP-Flow paper may prove instructive, showing 3 different instances of the 3 PNP steps:

{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/pnp_flow_steps.png" class="img-fluid" %}


This has a superficial resemblance to the "[ping-pong](https://github.com/Stability-AI/stable-audio-tools/blob/31932349d98c550c48711e7a5a40b24aa3d7c509/stable_audio_tools/inference/sampling.py#L221)" integration method used by the flow model Stable Audio Open Small (SAOS) <d-cite key="sao_small"></d-cite>, with a key distinction: the ping-pong integrator and updates the time-integrated latent variable $z$ (called "$x$" in SAOS), whereas for PnP-Flow it is the projection $\widehat{z_1}$ (called ["denoised"](https://github.com/Stability-AI/stable-audio-tools/blob/31932349d98c550c48711e7a5a40b24aa3d7c509/stable_audio_tools/inference/sampling.py#L242) in SAOS) that is the primary variable that is maintained between steps. This is a subtle distinction but worth noting.<d-footnote>The near trivial nature of integration for near-OT paths of flow models means that one can implement a flow version of DITTO <d-cite key="ditto"></d-cite> that avoids the expensive back-integration needed for diffusion models. The result is also superficially similar to PnP-Flow, but with gradient steps applied to $z$ instead of $\widehat{z_1}$.</d-footnote>

To implement PnP-Flow in code, let's replace our "integrator" with something specific to PnP-Flow:


{% highlight python %}
@torch.no_grad()
def sample_pnpflow(model, z0, n_steps=10, alpha=0.5, n_avg=5, warp_fn=lambda x: x, guidance_dict=None):
    ts = warp_fn(torch.linspace(0, 1, n_steps, device=z0.device, dtype=z0.dtype))  
    z1_hat = z0
    for t in ts[1:]:
        grad = guidance_dict['M_sq'] * (z1_hat - guidance_dict['y'])
        gamma_t = (1 - t) ** alpha
        z1_star = z1_hat - guidance_dict['strength'] * gamma_t * grad        
        projections = []           # Average multiple noisy projections
        for _ in range(n_avg):
            z = t * z1_star  +  (1 - t) * torch.randn_like(z1_star) 
            projections.append(z + (1 - t) * model(z, t))
        z1_hat = torch.stack(projections).mean(dim=0)
    return z1_hat
{% endhighlight %}


The model we used earlier for latents-only inpainting was trained to have straight trajectories, so we should be able to use it again here, just calling `sample_pnpflow` (instead of `integrate_path`). The results are as follows:

<details><summary>Show Code</summary>
{% highlight python %}
#| code-fold: true
@torch.no_grad()
def pnp_flow_inpaint(sub, inpainting_dict, n_samples, n_steps=10, t0=0, seed=None, **kwargs):
    """Inpaint using PnP-Flow method"""
    if seed is not None: torch.manual_seed(seed)
    z_y, M_sq = prepare_latent_mask(y, M, sub.encode)
    inpainting_dict.update({'M_sq': M_sq, 'y': z_y, 't_min': t0})
    z0 = torch.randn_like(z_y) * (1-t0) + z_y * t0
    return sample_pnpflow(sub.flow_model, z0, guidance_dict=inpainting_dict, **kwargs)


inpainting_dict = {'strength': 1.0, 't_min': 0.0, 't_max': 0.999}

show_grid(y,"Masked pixel images (y)")

z1 = pnp_flow_inpaint(sub, inpainting_dict, n_samples=len(y), n_steps=20, alpha=0.5, seed=0)
x1 = F.sigmoid(sub.decode(z1)).cpu()   # convert latents to pixels
show_grid(x1, f"Inpainted images (alpha=0.5, strength={inpainting_dict['strength']})")
{% endhighlight %}
</details>
{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/pnpflow_masked_ys.png" class="img-fluid" %}
{% include figure.liquid path="assets/img/2026-04-27-flow-where-you-want/pnpflow_inpainted_images.png" class="img-fluid" %}

Thus we see that works, though it also seems to "take some liberties": most of the letters in the second group look like "boldface" versions of the top ones. This could be because of the low spatial resolution of the latents, e.g. that they are encoding information about curvature or other aspects of the shape.

The results are similar to the previous inpainting method of Pokle et al, just a different way of doing it that may prove worthwhile. 

PnP-Flow is a general guidance method not limited to inpainting or even "linear" image degradations. I recommend looking into it further for other tasks and datasets --- and let me know what sort of results you find!



# Summary

Training generative models can be expensive—lots of data, electricity, compute time. So what if you could take a pretrained model and add controls at inference time instead? That's what this tutorial explored. While similar ideas are emerging for steering autoregressive models <d-cite key="zhao2025steeringautoregressive"></d-cite>, here we focused on pretrained flow models.

The key idea is simple: at each integration step, you project forward to estimate where you'll end up ($\,\widehat{z_1}\,$)
, check how far that is from where you want to be, and add a small velocity correction to steer toward your goal. We applied this to an unconditional MNIST flow model for two tasks: generating specific digit classes via classifier guidance, and filling in masked-out regions via inpainting.

We looked at four approaches. First, standard classifier guidance in pixel space—it works but it's slow because you're propagating gradients through the VAE decoder. Second, we trained a simple latent-space classifier and did the same thing much faster. Third, we implemented the linear inpainting method from Pokle et al, which operates directly on latents. Fourth, we tried PnP-Flow, which achieves guidance not by correcting velocities but by iteratively projecting samples forward and backward in time.

The math here is much simpler than for typical diffusion methods because flow trajectories are smooth and deterministic. We've glossed over a lot of detail compared to the research papers, but hopefully this gives you enough to experiment with your own controls. There are limits to the effectiveness of guidance: small models that don't generalize well won't suddenly work miracles if you try to push them too far outside their training distribution. Nevertheless, these plugin methods are worth exploring as accessible ways to steer generative flows where you want them to go. 
