# 📚 SwinLLIE Documentation

Complete documentation for the SwinLLIE (Swin Transformer for Low-Light Image Enhancement) project.

---

## 📖 Quick Navigation

| Document | Description | Audience |
|----------|-------------|----------|
| [THEORY_GUIDE.md](THEORY_GUIDE.md) | ⭐ **Start here!** Beginner-friendly theory | New users |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture details | Developers |
| [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md) | Custom dataset training | Researchers |
| [SWINIR_COMPARISON.md](SWINIR_COMPARISON.md) | Comparison with original SwinIR | Researchers |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Deploy to HuggingFace/Edge devices | Production |
| [API_REFERENCE.md](API_REFERENCE.md) | Code API documentation | Developers |
| [RESEARCH_PAPER_SWINLLIE.tex](RESEARCH_PAPER_SWINLLIE.tex) | LaTeX paper for Overleaf | Publication |

---

## 🎯 Learning Path

### For Beginners
1. Read [THEORY_GUIDE.md](THEORY_GUIDE.md) to understand the concepts
2. Run the quick start in the main [README](../README.md)
3. Explore [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

### For Researchers
1. Review [ARCHITECTURE.md](ARCHITECTURE.md) for implementation
2. Check [SWINIR_COMPARISON.md](SWINIR_COMPARISON.md) for differences from base model
3. Use [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md) for custom datasets
4. Reference [RESEARCH_PAPER_SWINLLIE.tex](RESEARCH_PAPER_SWINLLIE.tex) for publication

### For Production
1. Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment
2. Use [API_REFERENCE.md](API_REFERENCE.md) for integration

---

## 📊 Model Summary

| Aspect | Value |
|--------|-------|
| **Architecture** | U-Net Encoder-Decoder with Swin Transformer |
| **Parameters** | ~4.7M |
| **Input Size** | Any (auto-padded to multiple of 32) |
| **Window Size** | 8×8 (configurable) |
| **Loss Functions** | L1 + VGG + Color + Edge + Exposure |
| **Dataset** | LOL (Low-Light) Dataset |

---

## 📝 Citation

```bibtex
@article{swinllie2025,
  title={SwinLLIE: Swin Transformer for Low-Light Image Enhancement},
  author={Kavinda Mihiran},
  year={2025}
}
```

---

## 🔗 Related Resources

- [Original SwinIR Paper](https://arxiv.org/abs/2108.10257)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/)
