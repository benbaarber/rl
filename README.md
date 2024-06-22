# rl - A rust reinforcement learning library

[![Current Crates.io Version](https://img.shields.io/crates/v/rl.svg)](https://crates.io/crates/rl)
[![Documentation](https://img.shields.io/badge/Docs-latest-blue)](https://docs.rs/rl/0.4.0/rl/)
[![Rust Version](https://img.shields.io/badge/Rust-v1.79.0+-tan)](https://releases.rs/docs/1.79.0)

## About
**rl** is a fully Rust-native reinforcement learning library with the goal of providing a unified RL development experience, aiming to do for RL what libraries like PyTorch did for deep learning. By leveraging Rust's powerful type system and the [**burn**](https://github.com/tracel-ai/burn) library, **rl** enables users to reuse production-ready SoTA algorithms with arbitrary environments, state spaces, and action spaces. 

This project also aims to provide a clean platform for experimentation with new RL algorithms. By combining **burn**'s powerful deep learning features with **rl**'s provided RL sub-algorithms and components, users can create, test, and benchmark their own new experimental agents without having to start from scratch.

Currently, **rl** is in its early stages. Contributors are more than welcome!

## Features
 - High-performance production-ready implementations of all SoTA RL algorithms
 - Detailed logging and training visualization TUI (see image below)
 - Maximum extensibility for creating and testing new experimental algorithms
 - Gym environments
 - A comfortable learning experience for those new to RL
 - General RL peripherals and utility functions

![image](https://github.com/benbaarber/rl/assets/6320364/d0c545bb-a5f4-4487-8e33-1a02a3fb4577)
