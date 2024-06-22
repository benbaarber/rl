# rl - A rust reinforcement learning library

[![Current Crates.io Version](https://img.shields.io/crates/v/rl.svg)](https://crates.io/crates/rl)
[![Documentation](https://img.shields.io/badge/Docs-latest-blue)](https://docs.rs/rl/0.4.0/rl/)
[![Rust Version](https://img.shields.io/badge/Rust-v1.79.0+-tan)](https://releases.rs/docs/1.79.0)

## About
**rl** is a fully Rust-native reinforcement learning library with the goal of providing a unified RL development experience. This library is distinguished from other RL libraries in that
it leverages Rust's powerful type system to enable users to reuse the provided production-ready implementations with arbitrary environments, state spaces, and action spaces through generics. 
Libraries like PyTorch, Tensorflow, and Burn make implementing deep learning models easy, but implementing RL has always been a challenge. With **rl**, implementing RL in your project is just as easy.

The other goal of this project is to provide a clean platform for experimentation with new RL algorithms, including benchmarking with existing SoTA algorithms. By exposing all the internal sub-algorithms and components of common RL algorithms,
**rl** allows users to create new experimental agents without having to start from scratch. 

Currently, **rl** is in its early stages. Contributors are more than welcome!

## Features
 - High-performance production-ready implementations of all SoTA RL algorithms powered by the rust-native deep learning framework [burn](https://github.com/tracel-ai/burn)
 - Detailed logging and training visualization TUI (see image below)
 - Maximum extensibility for creating and testing new experimental algorithms
 - Gym environments
 - A comfortable learning experience for those new to RL
 - General RL peripherals and utility functions

![image](https://github.com/benbaarber/rl/assets/6320364/d0c545bb-a5f4-4487-8e33-1a02a3fb4577)
