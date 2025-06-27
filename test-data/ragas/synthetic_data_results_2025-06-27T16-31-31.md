```markdown
🚀 Ragas Synthetic Data Generation Demo
============================================================

1. Loading Documents
------------------------------
❌ Error loading documents: Directory not found: 'Sample_Docs_Markdown/'
Creating sample documents instead...
✅ Created 3 sample documents

2. Simple Testset Generation
------------------------------
🚀 Generating simple testset with 5 samples...
✅ Generated testset with 6 samples

3. Simple Testset Analysis
------------------------------

📝 Sample Queries (showing 3 of 6):
================================================================================

Query 1:
Question: N/A
Ground Truth: N/A
----------------------------------------

Query 2:
Question: N/A
Ground Truth: N/A
----------------------------------------

Query 3:
Question: N/A
Ground Truth: N/A
----------------------------------------

4. Knowledge Graph Testset Generation
------------------------------
🚀 Generating testset with 5 samples...
🔧 Creating Knowledge Graph...
Initial KnowledgeGraph: KnowledgeGraph(nodes: 0, relationships: 0)
KnowledgeGraph after adding documents: KnowledgeGraph(nodes: 3, relationships: 0)
🔄 Applying transformations to enrich knowledge graph...
Enriched KnowledgeGraph: KnowledgeGraph(nodes: 3, relationships: 3)
✅ Knowledge graph saved to knowledge_graph.json
Query distribution: [(SingleHopSpecificQuerySynthesizer(name='single_hop_specifc_query_synthesizer', llm=LangchainLLMWrapper(langchain_llm=ChatOpenAI(...)), generate_query_reference_prompt=QueryAnswerGenerationPrompt(instruction=Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) and the provided context. Ensure the answer is entirely faithful to the context, using only the information directly from the provided context.### Instructions:
1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question that aligns with the persona's perspective and incorporates the term.
2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer to the query. Do not add any information not included in or inferable from the context.
, examples=[(QueryCondition(persona=Persona(name='Software Engineer', role_description='Focuses on coding best practices and system design.'), term='microservices', query_style='Formal', query_length='Medium', context='Microservices are an architectural style where applications are structured as a collection of loosely coupled services. Each service is fine-grained and focuses on a single functionality.'), GeneratedQueryAnswer(query='What is the purpose of microservices in software architecture?', answer='Microservices are designed to structure applications as a collection of loosely coupled services, each focusing on a single functionality.'))], language=english), theme_persona_matching_prompt=ThemesPersonasMatchingPrompt(instruction=Given a list of themes and personas with their roles, associate each persona with relevant themes based on their role description., examples=[(ThemesPersonasInput(themes=['Empathy', 'Inclusivity', 'Remote work'], personas=[Persona(name='HR Manager', role_description='Focuses on inclusivity and employee support.'), Persona(name='Remote Team Lead', role_description='Manages remote team communication.')]), PersonaThemesMapping(mapping={'HR Manager': ['Inclusivity', 'Empathy'], 'Remote Team Lead': ['Remote work', 'Empathy']}))], language=english), property_name='entities'), 0.3333333333333333), (MultiHopAbstractQuerySynthesizer(name='multi_hop_abstract_query_synthesizer', llm=LangchainLLMWrapper(langchain_llm=ChatOpenAI(...)), generate_query_reference_prompt=QueryAnswerGenerationPrompt(instruction=Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) and the provided context. The themes represent a set of phrases either extracted or generated from the context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query explicitly incorporates these themes.### Instructions:
1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more themes and reflects their relevance to the context.
2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to the query. Avoid adding information that is not directly present or inferable from the given context.
3. **Multi-Hop Context Tags**:
   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.
   - Ensure the query uses information from at least two segments and connects them meaningfully., examples=[(QueryConditions(persona=Persona(name='Historian', role_description='Focuses on major scientific milestones and their global impact.'), themes=['Theory of Relativity', 'Experimental Validation'], query_style='Formal', query_length='Medium', context=['<1-hop> Albert Einstein developed the theory of relativity, introducing the concept of spacetime.', '<2-hop> The bending of light by gravity was confirmed during the 1919 solar eclipse, supporting Einstein’s theory.']), GeneratedQueryAnswer(query='How was the experimental validation of the theory of relativity achieved during the 1919 solar eclipse?', answer='The experimental validation of the theory of relativity was achieved during the 1919 solar eclipse by confirming the bending of light by gravity, which supported Einstein’s concept of spacetime as proposed in the theory.'))], language=english), concept_combination_prompt=ConceptCombinationPrompt(instruction=Form combinations by pairing concepts from at least two different lists.
**Instructions:**
- Review the concepts from each node.
- Identify concepts that can logically be connected or contrasted.
- Form combinations that involve concepts from different nodes.
- Each combination should include at least one concept from two or more nodes.
- List the combinations clearly and concisely.
- Do not repeat the same combination more than once., examples=[(ConceptsList(lists_of_concepts=[['Artificial intelligence', 'Automation'], ['Healthcare', 'Data privacy']], max_combinations=2), ConceptCombinations(combinations=[['Artificial intelligence', 'Healthcare'], ['Automation', 'Data privacy']]))], language=english), theme_persona_matching_prompt=ThemesPersonasMatchingPrompt(instruction=Given a list of themes and personas with their roles, associate each persona with relevant themes based on their role description., examples=[(ThemesPersonasInput(themes=['Empathy', 'Inclusivity', 'Remote work'], personas=[Persona(name='HR Manager', role_description='Focuses on inclusivity and employee support.'), Persona(name='Remote Team Lead', role_description='Manages remote team communication.')]), PersonaThemesMapping(mapping={'HR Manager': ['Inclusivity', 'Empathy'], 'Remote Team Lead': ['Remote work', 'Empathy']}))], language=english)), 0.3333333333333333), (MultiHopSpecificQuerySynthesizer(name='multi_hop_specific_query_synthesizer', llm=LangchainLLMWrapper(langchain_llm=ChatOpenAI(...)), generate_query_reference_prompt=QueryAnswerGenerationPrompt(instruction=Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) and the provided context. The themes represent a set of phrases either extracted or generated from the context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query explicitly incorporates these themes.### Instructions:
1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more themes and reflects their relevance to the context.
2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to the query. Avoid adding information that is not directly present or inferable from the given context.
3. **Multi-Hop Context Tags**:
   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.
   - Ensure the query uses information from at least two segments and connects them meaningfully., examples=[(QueryConditions(persona=Persona(name='Historian', role_description='Focuses on major scientific milestones and their global impact.'), themes=['Theory of Relativity', 'Experimental Validation'], query_style='Formal', query_length='Medium', context=['<1-hop> Albert Einstein developed the theory of relativity, introducing the concept of spacetime.', '<2-hop> The bending of light by gravity was confirmed during the 1919 solar eclipse, supporting Einstein’s theory.']), GeneratedQueryAnswer(query='How was the experimental validation of the theory of relativity achieved during the 1919 solar eclipse?', answer='The experimental validation of the theory of relativity was achieved during the 1919 solar eclipse by confirming the bending of light by gravity, which supported Einstein’s concept of spacetime as proposed in the theory.'))], language=english), relation_type='entities_overlap', property_name='entities', theme_persona_matching_prompt=ThemesPersonasMatchingPrompt(instruction=Given a list of themes and personas with their roles, associate each persona with relevant themes based on their role description., examples=[(ThemesPersonasInput(themes=['Empathy', 'Inclusivity', 'Remote work'], personas=[Persona(name='HR Manager', role_description='Focuses on inclusivity and employee support.'), Persona(name='Remote Team Lead', role_description='Manages remote team communication.')]), PersonaThemesMapping(mapping={'HR Manager': ['Inclusivity', 'Empathy'], 'Remote Team Lead': ['Remote work', 'Empathy']}))], language=english)), 0.3333333333333333)]
❌ Error during generation: No clusters found in the knowledge graph. Try changing the relationship condition.

```