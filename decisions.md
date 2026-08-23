# Decisiones técnicas del MVP

## 1. Alcance

El objetivo del MVP es demostrar que el asistente puede resolver preguntas sobre reglas, buscar cartas y analizar interacciones combinando el reglamento oficial con la API de Magic.

Se ha priorizado una solución sencilla que pueda ejecutarse localmente y explicarse con facilidad. La interfaz es una CLI porque permite probar el flujo completo sin dedicar tiempo a desarrollar una interfaz web.

## 2. LangChain y LangGraph

Se utiliza LangChain para construir los prompts y comunicarse con los modelos de Ollama. LangGraph se utiliza para organizar el flujo como un conjunto de nodos conectados.

El grafo permite definir de forma visible qué pasos sigue cada tipo de pregunta. Se ha preferido este flujo controlado frente a un agente con libertad para elegir herramientas, ya que las rutas necesarias son conocidas y así es más sencillo seguir y probar su comportamiento.

## 3. Clasificación de las preguntas

Después de preparar y reescribir la consulta, el clasificador selecciona una de estas rutas:

- pregunta de reglas;
- búsqueda de cartas;
- interacción entre cartas;
- creación de una carta personalizada;
- pregunta fuera de alcance.

Esta clasificación evita ejecutar siempre todos los componentes. Por ejemplo, una pregunta general sobre las fases de un turno no necesita consultar la API de cartas.

## 4. Preparación e historial de la conversación

LangGraph mantiene los mensajes mediante un checkpointer de SQLite y un identificador de conversación. SQLite es suficiente para una demo local con un solo proceso y no requiere desplegar una base de datos adicional.

Antes de clasificar cada nueva consulta, `RewriteQuery` utiliza el historial para convertir preguntas como "¿y qué ocurre con esa carta?" en consultas que puedan entenderse por sí solas. De esta forma, el resto de los nodos no necesita interpretar directamente toda la conversación.

## 5. Parseo y división del reglamento

El PDF se procesa siguiendo su estructura: capítulo, sección, regla y subregla. Se crea un chunk por regla completa y las subreglas permanecen dentro del mismo bloque.

Esta división conserva juntas las partes relacionadas de una regla y evita cortar el contenido utilizando un número fijo de caracteres. Cada chunk incluye metadatos como el número de regla, capítulo, sección, páginas y documento de origen para poder identificar la fuente recuperada.

## 6. Embeddings y Chroma

Los embeddings se generan localmente con `BAAI/bge-m3`. Se eligió porque permite representar consultas y reglas sin depender de una API de embeddings externa.

Los vectores se guardan en Chroma mediante almacenamiento persistente. Chroma permite construir y consultar un índice local con poca configuración, por lo que resulta adecuado para el alcance del MVP. El índice no se guarda en Git porque puede regenerarse a partir del PDF y del JSONL incluido en el repositorio.

## 7. Búsqueda de cartas

Para una búsqueda en lenguaje natural, un LLM extrae filtros estructurados como color, tipo, subtipo, texto o coste de maná. Una función Python utiliza después esos filtros para consultar la API de Magic.

La llamada a la API no la decide ni la ejecuta libremente un agente. Separar la interpretación del lenguaje y la consulta externa hace que los parámetros enviados sean visibles y que la función pueda probarse sin llamar al LLM.

## 8. Interacciones entre cartas

En una pregunta de interacción, el LLM identifica los nombres de las cartas y una función consulta sus datos en la API. El texto recuperado de las cartas se añade a la pregunta antes de buscar las reglas relacionadas en Chroma.

La respuesta final combina así dos fuentes: el texto de las cartas y el reglamento. Esta estructura permite justificar la respuesta aunque el modelo no conozca previamente la interacción concreta.

## 9. Generación de la respuesta

Un único nodo redacta la respuesta final utilizando la pregunta, las reglas recuperadas y los datos de las cartas. El prompt le indica que no invente información y que reconozca cuando el contexto no sea suficiente.

Centralizar la redacción evita crear una forma distinta de respuesta para cada ruta y mantiene separadas la recuperación de información y la generación del texto.

## 10. Creación de cartas personalizadas

La creación de cartas es un bonus y no se ha conectado al grafo principal. Se incluye un prototipo independiente que genera una ilustración con `x/z-image-turbo`.

El prototipo genera solo la imagen, sin intentar escribir mediante el modelo el nombre y las reglas de la carta, porque el texto generado dentro de una imagen puede resultar incorrecto. Una implementación completa necesitaría generar primero los campos de la carta y renderizarlos de forma controlada.

## 11. Tests

Los tests se centran en las partes deterministas: validación de chunks, preparación del turno, construcción de consultas a la API y preparación de interacciones.

No se realizan llamadas reales a Ollama ni a la API de Magic durante los tests. Esto permite que sean rápidos y repetibles y evita que fallen por cambios en la respuesta de un modelo o por problemas de conexión.

## 12. Limitaciones aceptadas

Por tratarse de un MVP, se han dejado fuera una interfaz web, autenticación, reintentos, caché, monitorización y recuperación de conversaciones anteriores desde la CLI. Estas necesidades se desarrollan por separado en la propuesta de evolución a producción.
