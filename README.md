# SmartGrid-ES

Proyecto académico de **aprendizaje por refuerzo con Gymnasium** para gestionar una **micro-red eléctrica inspirada en el sistema energético español**.

El objetivo es entrenar un agente capaz de tomar decisiones inteligentes sobre generación, almacenamiento, compra y venta de energía en un entorno estocástico, teniendo en cuenta factores como la producción renovable, la demanda, la climatología y el uso de fuentes de respaldo.

---

## Objetivo del proyecto

Desarrollar un entorno de simulación en Python usando **Gymnasium** en el que un agente de **Reinforcement Learning (RL)** aprenda a gestionar una micro-red compuesta por distintas fuentes de energía y sistemas de almacenamiento.

El sistema estará inspirado en elementos reales del mix energético español, incluyendo:

- Energía solar
- Energía eólica
- Hidráulica / bombeo
- Baterías modulares
- Ciclos combinados
- Nuclear
- Demanda variable
- Compra y venta de energía al mercado

El agente deberá aprender una política que permita:

- Cubrir la demanda energética
- Maximizar el uso de renovables
- Minimizar costes
- Evitar colapsos o déficits de suministro
- Gestionar correctamente las reservas energéticas

---

## Características principales

- Entorno desarrollado con **Gymnasium**
- Entrenamiento de agentes mediante **aprendizaje por refuerzo**
- Simulación de variables climatológicas:
  - radiación solar
  - viento
  - nubosidad
  - estaciones del año
- Inclusión de sistemas de almacenamiento:
  - baterías
  - bombeo hidráulico
- Fuentes de respaldo realistas
- Entorno estocástico con incertidumbre en generación, precios y demanda
- Comparación entre distintas versiones del proyecto:
  1. entorno prediseñado de Gymnasium
  2. versión con wrappers
  3. entorno propio
