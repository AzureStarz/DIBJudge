import re

text = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
Un groupe d'étudiants internationaux prévoit d'organiser une soirée multiculturelle dans une université française pour célébrer la diversité. La soirée comprendra des présentations culturelles, de la nourriture traditionnelle et des performances artistiques. Quelles seraient les meilleures recommandations pour organiser cet événement de manière respectueuse et inclusive, en tenant compte des différentes sensibilités culturelles des participants venant d'Asie, d'Afrique, d'Europe et des Amériques?

###Response A to evaluate:
Pour organiser la soirée multiculturelle à l'université, il faudrait d'abord former une équipe d'organisation avec des étudiants de différentes origines pour planifier l'événement. Il serait important de prévoir un espace suffisamment grand pour accueillir tous les participants et leurs présentations culturelles. La nourriture devrait être étiquetée clairement pour indiquer les ingrédients principaux, et il faudrait prévoir des options pour différents régimes alimentaires. Les présentations culturelles pourraient inclure des danses traditionnelles, de la musique et des démonstrations d'art de chaque pays représenté. Il serait utile d'avoir des panneaux d'information en plusieurs langues pour que tout le monde puisse comprendre le programme de la soirée. Pour la décoration, on pourrait utiliser des drapeaux et des symboles des différents pays participants. Les participants devraient être encouragés à porter leurs tenues traditionnelles s'ils le souhaitent. Une bonne communication entre les organisateurs et les participants sera essentielle pour assurer le bon déroulement de l'événement.

###Response B to evaluate:
Pour assurer le succès de cette soirée multiculturelle, je recommande de former un comité organisateur diversifié représentant chaque région participante, permettant ainsi une planification équilibrée et inclusive. La présentation des plats traditionnels devrait être accompagnée d'explications détaillées dans plusieurs langues, respectant les diverses restrictions alimentaires et préférences culturelles. Il serait judicieux d'organiser des consultations préalables avec des représentants de chaque communauté pour valider la pertinence et l'authenticité des performances artistiques prévues. L'aménagement de l'espace devrait tenir compte des différentes normes culturelles, incluant des zones adaptées pour la prière, la méditation, et des espaces permettant des interactions confortables pour tous. Une équipe multilingue devrait être disponible pendant l'événement pour faciliter la communication et résoudre rapidement tout malentendu culturel potentiel. Des directives claires concernant la photographie et le partage sur les réseaux sociaux devraient être établies, respectant les sensibilités de chaque groupe culturel. Un système de feedback devrait être mis en place pour recueillir les impressions des participants après l'événement, permettant d'améliorer les futures éditions. Cette approche garantira une célébration respectueuse et enrichissante de la diversité culturelle, favorisant les échanges authentiques entre les participants.

###Evaluation Criteria:
Does the model's feedback reflect a profound comprehension of cultural subtleties and considerations?

###Feedback:
Both responses demonstrate an understanding of organizing a multicultural event, but there are notable differences in their depth of cultural considerations. Response A provides basic organizational elements like venue, food labeling, and cultural presentations, focusing primarily on logistical aspects. While it mentions important elements like multilingual information panels and traditional attire, it lacks deeper consideration of cultural sensitivities and potential challenges.

Response B exhibits a more nuanced understanding of cultural dynamics by suggesting preliminary consultations with community representatives to validate the authenticity of performances. It addresses crucial cultural considerations often overlooked, such as providing prayer and meditation spaces, establishing guidelines for photography and social media sharing, and having a multilingual team to handle potential cultural misunderstandings. The recommendation to form a diverse organizing committee representing each participating region demonstrates a more thoughtful approach to ensuring cultural representation from the planning stage.

The suggestion in Response B to implement a feedback system shows a commitment to continuous improvement and cultural learning. Additionally, its emphasis on detailed food explanations in multiple languages and consideration of various cultural norms regarding space usage reflects a deeper understanding of how different cultural backgrounds might affect participants' comfort and participation.

While Response A provides a solid foundation for event organization, Response B is superior in its comprehensive approach to cultural sensitivity, showing a more profound understanding of the subtle cultural considerations necessary for creating a truly inclusive and respectful multicultural event. [RESULT] B
"""

pattern = re.compile(
    r"###Task Description:\s*(?P<task_description>[\s\S]*?)"
    r"###The instruction to evaluate:\s*(?P<instruction>[\s\S]*?)"
    r"###Response A to evaluate:\s*(?P<response_A>[\s\S]*?)"
    r"###Response B to evaluate:\s*(?P<response_B>[\s\S]*?)"
    r"###Evaluation Criteria:\s*(?P<evaluation_criteria>[\s\S]*?)"
    r"###Feedback:\s*(?P<feedback>[\s\S]*?)$"
)

match = pattern.search(text)

if not match:
    print("❌ Regex did not match")
else:
    print("✅ Regex matched successfully\n")
    for key, value in match.groupdict().items():
        print(f"===== {key.upper()} =====")
        print(value.strip())
        print()
