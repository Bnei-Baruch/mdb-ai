from urllib.request import urlopen

from flask import make_response

from summarization.run_model import run

TXT_HE = """
לכן צריכה החברה להיות כלולה מיחידים, שכולם בדיעה אחת, שצריכים להגיע לזה. אז מכל היחידים נעשה כח גדול אחד, שיכול להילחם עם עצמו, מטעם שכל אחד כלול מכולם. נמצא, שכל אחד הוא מיוסד על רצון גדול, שהוא רוצה להגיע להמטרה.

ובכדי שתהיה התכללות אחד מהשני, אז כל אחד צריך לבטל את עצמו נגד השני. וזהו על ידי זה שכל אחד רואה מעלות חבירו ולא חסרונו. אבל מי שחושב, שהוא קצת גבוה מהחברים, כבר הוא לא יכול להתאחד עמהם.

וכמו כן בעת ההתאספות, צריכים להיות רציניים, בכדי לא לצאת מהכוונה, שעל הכוונה זו נתאספו. ומטעם הצנע לכת, שזה ענין גדול מאוד, היו רגילים להראות מבחוץ, שהוא לא רציני. אבל באמת בתוך ליבם היה אש בוערת.

אבל לאנשים קטנים, על כל פנים בעת האסיפה, צריכים להיזהר, לא להמשיך אחרי דיבורים ומעשים, שלא מביאים את המטרה, שהתאספו, שהוא, שעל ידי זה צריכים להגיע לדביקות ה'. וענין דביקות עיין בספר "מתן תורה" (דף קס"ח, דברי המתחיל "ובאמת").

רק בזמן שנמצאים לא עם החברים, אז יותר טוב שלא להראות לחוץ את הכוונה שיש בליבם, ולהיות בחיצוניות כמו כולם, שזה סוד "והצנע לכת עם ה' אלקיך". הגם שיש על זה פירושים יותר גבוהים, אבל הפירוש הפשוט הוא גם כן ענין גדול.

לכן כדאי, שבין החברים, שמתחברים, תהיה להם השתוות אחד עם השני, בכדי שיוכל להיבטל אחד לפני השני. ובהחברה צריכה להיות שמירה יתרה, שלא יכנס בתוכם ענין של קלות ראש, משום שקלות ראש הורס את הכל. אבל כנ"ל, זה צריך להיות ענין פנימי.

אבל בזמן שנמצא מי שהוא, אם הוא אינו מחברה זו, צריכים לא להראות שום רצינות, אלא להשתוות מבחוץ עם אדם שבא עכשיו. היינו, שלא לדבר מדברים רציניים, אלא מדברים שמתאימים לאדם שבא עכשיו. שהוא נקרא "אורח בלתי קרוא".
"""

TXT_EN = """
 We have gathered here to establish a society for all who wish to follow the path and method of Baal HaSulam, the way by which to climb the degrees of man and not remain as a beast, as our sages said (Yevamot, 61a) about the verse, “And you My sheep, the sheep of My pasture, are men.” And Rashbi said, “You are called ‘men,’ and idol worshipers are not called ‘men.’”

To understand man’s merit, we shall now bring a verse from our sages (Berachot, 6b) about the verse, “The end of the matter, all having been heard: fear God, and keep His commandments; for this is the whole man” (Ecclesiastes, 12:13). And the Gemarah asks, “What is ‘for this is the whole man’”?

Rabbi Elazar said, “The Creator said, ‘The whole world was created only for that.’ This means that the whole world was created for the fear of God.”

Yet, we need to understand what the fear of God is, being the reason for which the world was created. From all the words of our sages, we learn that the reason for creation was to benefit His creations. This means that the Creator wished to delight the creatures so they would feel happy in the world. And here our sages said about the verse, “For this is the whole man,” that the reason for creation was the fear of God.

But according to what is explained in the essay, Matan Torah [“The Giving of the Torah”], the reason why the creatures are not receiving delight and pleasure, even though it was the reason for creation, is the disparity of form between the Creator and the creatures. The Creator is the giver and the creatures are the receivers. But there is a rule that the branches are similar to the root from which the branches were born.
"""

TXT_RU = """
Мы собрались здесь, чтобы заложить основу здания группы для всех тех, кто заинтересован идти по пути и методике Бааль Сулама, по пути, [указывающему,] как подняться по ступеням человека, а не оставаться в состоянии животного, как сказали наши мудрецы о стихе: ««И вы – овцы Мои, овцы паствы Моей, вы есть человек»1 – вы называетесь человеком, а идолопоклонники не называются человеком»2, что является высказыванием Рашби.

И чтобы понять, что такое ступень человека, приведем здесь высказывание наших мудрецов о стихе: «Послушаем всему заключенье: пред Творцом трепещи и заповеди Его соблюдай, ибо в этом весь человек»3. И спрашивает Гмара: «Что значит: «Ибо в этом весь человек»? Сказал рабби Элазар: «Сказал Творец: «Весь мир создан только ради этого»»»4, – что означает, что весь мир создан только ради трепета перед Творцом.

И следует понять, что такое «трепет перед Творцом», ведь получается, что это является причиной, по которой был создан мир. И известно из всех высказываний наших мудрецов, что причиной творения было [желание] насладить Свои создания. Т.е. Творец хотел насладить создания, чтобы они чувствовали себя счастливыми в мире. А здесь мудрецы сказали о стихе: «Ибо в этом весь человек», что причина творения есть «трепет перед Творцом».
"""
def handle_summary():
    # lang = request.values['lang']
    # if 'txt' not in request.files:
    #   return Response('must send or file', status=422)
    url = "https://kabbalahmedia.info/assets/api/doc2html/8IeKE5XQ"
    res = urlopen(url)
    try:
        rez = run(TXT_EN)
    except Exception as e:
        return make_response(str(e), 500)
    return make_response(rez, 200)
