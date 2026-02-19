# TFE: Optimización de Stock para un Modelo de Negocio Multicanal
Este repositorio contiene el código fuente y la arquitectura técnica desarrollada para el Trabajo de Fin de Máster (TFE) enfocado en la transición de datos manuales a decisiones inteligentes de inventario.

# Descripción del Proyecto
El sistema implementa un motor de decisiones prescriptivas que utiliza Machine Learning (XGBoost) para predecir la demanda y generar alertas automáticas de rotura de stock. La solución está integrada en un ecosistema serverless en Google Cloud Platform, permitiendo una escalabilidad total y una visualización en tiempo real a través de Looker Studio.

# Estructura del Repositorio
La organización del código sigue las fases detalladas en la memoria del proyecto:
1. 01_Prototipado/: Cuadernos de Jupyter con el Análisis Exploratorio de Datos (EDA) y el entrenamiento inicial del modelo en Google Colab.
2. 02_produccion/: Código fuente refactorizado (main.py) para la ejecución automatizada y conexión vía API XML-RPC con Odoo.
3. 03_docker/: Configuración de la contenedorización para el despliegue en Google Cloud Run.
requirements.txt: Listado de librerías y dependencias necesarias para el entorno de ejecución.

# Stack Tecnológico
1. Lenguaje: Python (Pandas, Scikit-learn, XGBoost, PyArrow).
2. Infraestructura: Google Cloud Platform (BigQuery, Cloud Run, Artifact Registry).
3. Contenedorización: Docker.
4. Visualización: Looker Studio.

# Lógica de Negocio Implementada
El motor procesa variables críticas para la toma de decisiones, tales como:
1. Stock Proyectado: Cálculo basado en el stock actual más las reposiciones inminentes (ventas_7d_totales).
2. Cooling Logic: Algoritmo de post-procesamiento para eliminar falsos positivos en artículos sin rotación reciente.
3. Semáforo de Riesgo: Clasificación binaria (alarma_critica) para la gestión prioritaria de almacén.

Nota de Seguridad: Las credenciales de acceso a las APIs y servicios de GCP se gestionan de forma segura mediante variables de entorno en Cloud Run y no están incluidas en este repositorio para garantizar la integridad del sistema.
