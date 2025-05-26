# Galaxy Morphology Generation with Continuous Redshift-# Galaxy Morphology Generation with Continuous Redshift-Conditioned DDPMs

A breakthrough implementation of Denoising Diffusion Probabilistic Models (DDPMs) that generates scientifically accurate galaxy images across cosmic time, establishing a direct link between galaxy morphology and redshift using continuous conditioning—the first systematic evaluation of continuous galaxy evolution with DDPMs.

## 🌟 Project Overview

This project addresses a fundamental challenge in observational astronomy: spectroscopic redshift measurements require ~1000x more telescope time than photometric imaging, severely limiting sample sizes. By training a conditional DDPM on the Hyper Suprime-Cam (HSC) survey data, we demonstrate that **redshift alone is sufficient for a generative model to implicitly learn complex morphological properties** without explicit supervision.

Our model learns the conditional distribution p(X|z) of 5-band galaxy images given continuous redshift values (z=0 to z=4), capturing morphological evolution across 12+ billion years of cosmic time.

### Key Achievements
- **First morphology-redshift link**: Computational evidence that galaxy structure encodes redshift information
- **Implicit physical learning**: Learns ellipticity, size, and profile parameters without morphological labels  
- **Continuous conditioning**: Overcomes information loss from discrete redshift binning
- **Statistical validation**: Generated galaxies match real morphological distributions (>90% accuracy)

## 🏗️ Technical Architecture

### Continuous Redshift Conditioning
**Core Innovation**: Unlike previous approaches that discretize redshift (losing evolutionary information), we integrate continuous redshift directly into the DDPM framework with Gaussian perturbation for smooth interpolation.

```python
# Training: Redshift conditioning with label noise for robustness
labels += torch.randn_like(labels) * 0.01  # Small noise addition
labels = torch.clamp(labels, 0, 4)

# Forward diffusion process  
q(X_t|X_{t-1}) = N(X_t; √(1-β_t)X_{t-1}, β_t I)

# Learned denoising with redshift conditioning
ε_θ(X_t, t, z) → predicted noise
```

### Model Architecture

**Conditional U-Net Design**:
```python
class UNet_conditional_conv(nn.Module):
    def __init__(self, c_in=5, c_out=5, time_dim=256, y_dim=1):
        # 5-channel input/output for g,r,i,z,y bands
        # Continuous redshift conditioning via linear embedding
        self.label_emb = nn.Linear(y_dim, time_dim)
        
    def forward(self, x, t, z):
        # Sinusoidal positional encoding for timestep
        t_emb = self.pos_encoding(t, self.time_dim)
        # Add redshift embedding to time encoding
        if z is not None:
            z_emb = self.label_emb(z)
            t_emb = t_emb + z_emb
```

**Key Components**:
- **Input**: 64×64×5 multi-band astronomical images (g,r,i,z,y filters)
- **Time Embedding**: 256-dim sinusoidal encoding for diffusion timesteps
- **Redshift Integration**: Linear embedding added to time encoding for continuous conditioning
- **Architecture**: U-Net with residual blocks, group normalization, GELU activation
- **Stabilization**: EMA model (β=0.995) for consistent high-quality generation

### Training Pipeline

**Diffusion Process**:
- **Noise Schedule**: Linear β from 1e-4 to 0.02 over 1000 timesteps
- **Loss Function**: Huber Loss (smooth_l1_loss) for robustness to astronomical outliers
- **Optimizer**: AdamW with lr=5e-5, gradient clipping (max_norm=1.0)
- **Batch Size**: 128 on single NVIDIA A6000 GPU

**Data Pipeline**:
```python
# HDF5 efficient loading with custom dataset class
class HDF5ImageGenerator(Dataset):
    def __init__(self, src, X_key='image', y_key='specz_redshift'):
        # Handles 5-channel astronomical data
        # Continuous redshift labels (no discretization)
        # Configurable normalization for astronomical ranges
        
# Training data flow
images = normalize_images(images)  # [-1, 1] normalization
t = diffusion.sample_timesteps(batch_size)
x_t, noise = diffusion.noise_images(images, t)
predicted_noise = model(x_t, t, redshifts)
loss = F.smooth_l1_loss(noise, predicted_noise)
```

**Advanced Training Features**:
- **Automatic Checkpointing**: Save state every 2 epochs with cleanup
- **Validation Monitoring**: Track morphological accuracy on held-out data
- **TensorBoard Integration**: Real-time loss curves and generated sample visualization
- **Error Handling**: Comprehensive logging and recovery mechanisms

## 🔬 Scientific Applications

### 1. Galaxy Evolution Studies
**Generate evolutionary sequences** to study morphological changes across cosmic time:
```python
# Generate galaxy at multiple redshifts
redshifts = torch.linspace(0.1, 4.0, 20)
evolution_sequence = diffusion.sample(model, len(redshifts), redshifts)
```

**Demonstrated capabilities**:
- Learn morphological properties without explicit supervision
- Generate galaxies with realistic physical characteristics at any redshift
- Capture redshift-dependent trends in ellipticity, size, and brightness profiles

### 2. Astrophysical Research
**Validate morphology-redshift relationships**:
- Test theoretical predictions about galaxy evolution
- Study morphological parameter distributions across cosmic time
- Investigate correlations between redshift and structural properties

### 3. Machine Learning Applications  
**Data augmentation and algorithm development**:
- Generate synthetic training data for photometric redshift estimation
- Test morphological analysis algorithms on controlled datasets
- Validate image processing pipelines with known ground truth

## 📊 Evaluation & Validation

### Three-Notebook Workflow
The project provides **three easy-to-follow Jupyter notebooks**:
1. **Training Notebook**: Complete training pipeline with checkpointing and monitoring
2. **Generation Notebook**: Image generation with real vs synthetic comparisons  
3. **Evaluation Notebook**: Comprehensive morphological analysis and statistical validation

### Morphological Analysis Pipeline
**Quantitative validation** using Source Extraction and Photometry (SEP):

```python
def analyze_image_with_sep(image, redshift):
    # Background subtraction and source detection
    bkg = sep.Background(image)
    objects = sep.extract(image - bkg, threshold=1.5)
    
    # Extract morphological parameters
    return {
        'Semi-major Axis': obj['a'],
        'Semi-minor Axis': obj['b'], 
        'Ellipticity': 1 - obj['b']/obj['a'],
        'Orientation Angle': obj['theta'],
        'Isophotal Area': obj['npix'],
        'Sersic Index': calculated_sersic_index
    }
```

### Performance Metrics (Table 1 Results)
**Statistical consistency** with HSC observations (σ=0.1 optimal):

| Parameter | Generated vs Real Ratio |
|-----------|------------------------|
| Ellipticity | 0.98 |
| Semi-major Axis | 0.96 |
| Sérsic Index | 0.93 |
| Isophotal Area | 0.90 |
| **FID Score** | **14.2** |

> **Note**: Results shown are from optimized parameters as reported in the research paper. All evaluation parameters are configurable in the provided notebooks.

**Benchmark Comparison**:
- **vs Discrete DDPM**: ~2x better morphological accuracy (e.g., ellipticity: 0.98 vs 0.45)
- **vs Conditional GANs**: Significantly better physical parameter reproduction
- **vs Previous Methods**: Only approach to achieve >90% accuracy on all morphological metrics

### Redshift Evolution Validation
**Captures observed astrophysical trends** (Figures 3-4):
- **Size Evolution**: Galaxies appear smaller at higher redshift due to cosmological effects
- **Morphological Trends**: Proper evolution of ellipticity and structural parameters with redshift
- **Distribution Matching**: Generated galaxy populations statistically indistinguishable from real data

## 💻 Implementation Details

### Generation Process
```python
# Load trained model and generate galaxies
model = UNet_conditional_conv(c_in=5, c_out=5, y_dim=1)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate at specific redshifts
sample_redshifts = torch.tensor([[0.5], [1.0], [2.0], [3.0]])
generated_galaxies = diffusion.sample(ema_model, 4, sample_redshifts)

# Denormalize for visualization
galaxies_display = denormalize_images(generated_galaxies)
```

### Data Requirements
- **Input Format**: HDF5 files with 'image' (N,5,64,64) and 'specz_redshift' (N,) keys
- **Training Data**: 204,513 HSC galaxies with spectroscopic redshifts
- **Validation**: 40,914 independent test galaxies
- **Memory**: ~8GB GPU memory for batch_size=128

### Computational Performance
- **Training Time**: ~600 epochs (several days on A6000)
- **Generation Speed**: 64×64×5 galaxy in ~30 seconds
- **Scalability**: Efficient batch processing for mock catalog generation
- **Memory Optimization**: HDF5 streaming, gradient checkpointing

## 🎯 Future Research Directions

Future work should focus on:

### Physics Validation
- **Star Formation Rate Evolution**: Test if generated galaxies reproduce observed star formation rate density evolution across redshift
- **Galaxy Merger Physics**: Apply more stringent tests for physical changes through galaxy merger processes
- **Direct Physics Learning**: Extend approach toward models that can learn the physical evolution of galaxies more directly

### Dynamic Visualization
- **Evolutionary Trajectories**: Leverage DDPM's ability to interpolate between probability distribution modes for dynamic visualizations of galaxy evolution as a function of redshift
- **Cosmic Time Sequences**: Create continuous evolutionary sequences serving as a powerful tool for studying galaxy formation across cosmic timescales

## 📄 Research Paper

This implementation is based on research submitted to NeurIPS 2025. The paper demonstrates that continuous redshift conditioning enables DDPMs to implicitly learn complex morphological properties of galaxies without explicit supervision.

**Key Findings**:
- First direct link established between galaxy morphology and redshift using continuous conditioning
- Generated galaxies maintain >90% statistical accuracy with real morphological distributions
- Continuous conditioning significantly outperforms discrete redshift binning approaches

*ArXiv link will be available upon publication*

## 📊 Dataset

This project uses the **Hyper Suprime-Cam Galaxy Dataset** curated by Do et al., containing 286,401 galaxies with spectroscopic redshifts spanning z=0 to z=4.

**Dataset Details**:
- **Source**: HSC Survey - deepest completed wide-field survey
- **Coverage**: 5 optical bands (g, r, i, z, y) at 64×64 pixel resolution  
- **Depth**: r-band depth of ~26.5 AB magnitudes
- **Redshift Range**: z=0 to z=4 (92.8% have z<1.5)

**Access**: [HSC Galaxy Dataset on Zenodo](https://zenodo.org/records/11117528) (CC-BY 4.0 License)

---

This work demonstrates that state-of-the-art generative models can learn fundamental astrophysical relationships from data alone, providing both a powerful tool for astronomical research and new insights into the deep connections between galaxy structure and cosmic evolution.
