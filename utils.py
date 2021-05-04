from collections import Counter
import cv2
import numpy as np

def append_txt_file(s:str, directory:str) -> None:
    with open(directory, 'a') as f:
        f.write(s)

def get_txt_file(directory:str) -> list:
    with open(directory, 'r') as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines if (line[-1] == '\n')]
    return lines

class THC(object):
    def __init__(self, fullname:str, english_name:str, *alias):
        self.fullname = fullname
        self.english  = english_name
        self.alias    = tuple(alia.lower() for alia in alias)

class THS(object):
    def __init__(self, familyname:str, *sisters):
        self.familyname = familyname
        self.sisters    = sisters

class THClist(object):
    def __init__(self, thc_list:list, ths_list:list):
        self.thc_list = thc_list
        self.thc_checker = [[n.fullname, *list(n.alias)] for n in thc_list]
        self.flat_checker = [item for sublist in self.thc_checker for item in sublist]

        self.ths_list = ths_list

        self.alias_dict = {}
        for names in self.thc_checker:
            self.alias_dict[names[0]] = names[0]
            for name in names[1:]:
                self.alias_dict[name] = names[0]

        self.sister_dict = {n.familyname: n.sisters for n in self.ths_list}

    def get_characters(self, tags, ignore_unkown=True,
                       ちゃん抜き=True, hash抜き=True, ゆっくり抜き=True) -> list:
        dupes = []
        for tag in tags:
            if tag in self.sister_dict:
                dupes += self.sister_dict[tag]
            else:
                tag = tag.lower()
                if ちゃん抜き and (tag[-3:] == 'ちゃん'):
                    tag = tag[:-3]
                if hash抜き and (tag[0] == '#'):
                    tag = tag[1:]
                if hash抜き and (tag[:4] == 'ゆっくり'):
                    tag = tag[4:]
                dupes += [tag]
        dupes = list(set(dupes))
        dupes = [item for item, count in Counter(self.flat_checker + dupes + (dupes if not ignore_unkown else [])).items() if count > 1]
        dupes = list(set([(self.alias_dict[dupe] if (dupe in self.alias_dict) else dupe) for dupe in dupes]))
        return dupes

    def one_hot_encode(self, characters):
        output = [(character.fullname in characters) for character in self.thc_list]
        return output

    def one_hot_decode(self, pred, top=3, round_num=3, english=False):
        pred = np.apply_along_axis(self.translate_single, -1, pred, top=top, round_num=round_num, english=english)
        return pred

    def translate_single(self, array, top=3, round_num=3, english=False):
        if english:
            pred_dict = {character.english: 0.0 for character in self.thc_list}
        else:
            pred_dict = {character.fullname: 0.0 for character in self.thc_list}
        pred_list = array.tolist()
        for n in range(len(pred_list)):
            if english:
                pred_dict[self.thc_list[n].english] += pred_list[n]
            else:
                pred_dict[self.thc_list[n].fullname] += pred_list[n]
        pred_list = [(pred_dict[name], name) for name in pred_dict]
        pred_list.sort(reverse=True)
        if top is None:
            pred_list = [(prob, name) for prob, name in pred_list if (prob > 0.0)]
        else:
            pred_list = pred_list[:top]
        pred_dict = {name: round(score, round_num) for score, name in pred_list}
        return pred_dict



all_sisters = [
    THS('プリズムリバー三姉妹', 'ルナサ・プリズムリバー', 'メルラン・プリズムリバー', 'リリカ・プリズムリバー'),
    THS('依神姉妹', '依神女苑', '依神紫苑'),
    THS('スカーレット姉妹', 'レミリア・スカーレット', 'フランドール・スカーレット'),
    THS('秋姉妹', '秋穣子', '秋静葉'),
    THS('古明地姉妹', '古明地さとり', '古明地こいし'),
    THS('九十九姉妹', '九十九弁々', '九十九八橋'),

    THS('レイマリ', '博麗霊夢', '霧雨魔理沙'),
    THS('マリアリ', '霧雨魔理沙', 'アリス・マーガトロイド'),
    THS('さとこい', '古明地さとり', '古明地こいし'),
    THS('大チル', '大妖精', 'チルノ'),
    THS('勇パル', '星熊勇儀', '水橋パルスィ'),
    THS('蓮メリ', '宇佐見蓮子', 'マエリベリー・ハーン'),
    THS('あやれいむ', '射命丸文', '博麗霊夢'),
    THS('みまりさ', '魅魔', '霧雨魔理沙'),
    THS('レミフラ', 'レミリア・スカーレット', 'フランドール・スカーレット')

]

all_characters = [
    # PROTAGONISTS
    THC('博麗靈夢', 'Reimu Hakurei (PC98)', '靈夢'),
    THC('博麗霊夢', 'Reimu Hakurei', '霊夢', 'れいむ','Reimu'),
    THC('霧雨魔理沙', 'Marisa Kirisame', '魔理沙', 'まりさ', 'Marisa'),

    # HIGHLY RESPONSIVE TO PRAYERS
    THC('しんぎょく', 'SinGyoku'),
    THC('幽玄魔眼', 'YuugenMagan'),
    THC('エリス', 'Elis'),
    THC('サリエル', 'Sariel'),
    THC('魅魔', 'Mima', '魅魔様'),
    THC('キクリ', 'Kikuri'),
    THC('こんがら', 'Konngara', 'コンガラ'),

    # STORY OF EASTERN WONDERLAND
    THC('玄爺', 'Genjii'),
    THC('里香', 'Rika'),
    THC('明羅', 'Meira'),

    # PHANTASMAGORIA OF DIM. DREAM
    THC('エレン', 'Ellen'),
    THC('小兎姫', 'Kotohime'),
    THC('カナ・アナベラル', 'Kana Anaberal', 'カナ', 'カナアナベラル', 'カナ･アナベラル'),
    THC('朝倉理香子', 'Rikako Asakura','理香子'),

    THC('北白河ちゆり', 'Chiyuri Kitashirakawa'),
    THC('岡崎夢美', 'Yumemi Okazaki'),
    THC('る～こと', 'Ruukoto', 'るーこと'),

    # LOTUS LAND STORY
    THC('オレンジ', 'Orange'),
    THC('くるみ', 'Kurumi'),
    THC('エリー', 'Elly'),
    # THC('幽香'),
    THC('夢月', 'Mugetsu'),
    THC('幻月', 'Gengetsu'),

    # MYSTIC SQUARE
    THC('サラ', 'Sara'),
    THC('ルイズ', 'Louise'),
    # THC('アリス'),
    THC('ユキ', 'Yuki'),
    THC('マイ', 'Mai'),
    THC('夢子', 'Yumeko'),
    THC('神綺', 'Shinki'),

    # MISC CHARACTERS
    THC('上海人形', 'Shanghai Doll', '上海'),
    THC('蓬莱人形', 'Hourai Doll', '蓬莱'),
    THC('レイラ・プリズムリバー', 'Layla Prismriver', 'レイラ'),
    THC('朱鷺子', 'Tokiko', '名無しの本読み妖怪'),

    # MANGAS AND CDS
    THC('宇佐見蓮子', 'Renko Usami', '蓮子'),
    THC('マエリベリー・ハーン', 'Maribel Hearn', 'メリー', 'マエリベリー', 'マエリベリー･ハーン', 'マエリベリーハーン'),
    THC('森近霖之助', 'Rinnosuke Morichika', '霖之助'),
    THC('稗田阿求', 'Hieda no Akyuu', '稗田', '阿求'),
    THC('綿月豊姫', 'Watatsuki no Toyohime', '豊姫'),
    THC('綿月依姫', 'Watatsuki no Yorihime', '依姫'),
    THC('レイセン', 'Reisen'),
    THC('本居小鈴', 'Kosuzu Motoori', '小鈴'),
    THC('奥野田美宵', 'Miyoi Okunoda', '美宵'),

    # EMBODIMENT OF SCARLET DEVIL
    THC('ルーミア', 'Rumia', 'Rumia'),
    THC('大妖精', 'Daiyousei', '大ちゃん', 'Daiyousei'),
    THC('チルノ', 'Cirno', '琪露诺', 'Cirno'),
    THC('紅美鈴', 'Hong Meiling', '美鈴', '红美铃', 'Meiling'),
    THC('小悪魔', 'Koakuma', '小悪魔(東方Project)', 'Koakuma'),
    THC('パチュリー・ノーレッジ', 'Patchouli Knowledge', 'パチュリー･ノーレッジ', 'パチュリー', 'パチュリーノーレッジ', 'Patchouli'),
    THC('十六夜咲夜', 'Sakuya Izayoi', '咲夜', '咲夜さん', 'Sakuya'),
    THC('レミリア・スカーレット', 'Remilia Scarlet', 'レミリア', 'レミリアスカーレット', 'レミリア･スカーレット', 'Remilia'),
    THC('フランドール・スカーレット', 'Flandre Scarlet', 'フランドール', 'フランドールスカーレット', 'フランドール･スカーレット', 'フラン', 'Flandre'),

    # PREFECT CHERRY BLOSSOM
    THC('レティ・ホワイトロック', 'Letty Whiterock', 'レティ', 'レティホワイトロック', 'レティ･ホワイトロック', 'Letty'),
    THC('橙', 'Chen', 'ちぇん', '八雲橙', 'Chen'),
    THC('アリス・マーガトロイド', 'Alice Margatroid', 'アリス', 'アリス･マーガトロイド', 'アリスマーガトロイド', 'Alice'),
    THC('リリーホワイト', 'Lily White', 'リリー・ホワイト'),
    THC('ルナサ・プリズムリバー', 'Lunasa Prismriver', 'ルナサ', 'ルナサ･プリズムリバー', 'ルナサプリズムリバー'),
    THC('メルラン・プリズムリバー', 'Merlin Prismriver', 'メルラン', 'メルラン･プリズムリバー', 'メルランプリズムリバー'),
    THC('リリカ・プリズムリバー', 'Lyrica Prismriver', 'リリカ', 'リリカ･プリズムリバー', 'リリカプリズムリバー'),
    THC('魂魄妖夢', 'Youmu Kompaku','妖夢', 'みょん', 'youmu'),
    THC('西行寺幽々子', 'Yuyuko Saigyouji', '幽々子', '幽々子様', 'Yuyuko'),
    THC('八雲藍', 'Ran Yakumo', '藍', 'Ran'),
    THC('八雲紫', 'Yukari Yakumo','紫', '紫様', 'Yukari'),

    # IMMATERIAL AND MISSING POWER
    THC('伊吹萃香', 'Suika Ibuki', '萃香'),

    # IMPERISHABLE NIGHT
    THC('リグル・ナイトバグ', 'Wriggle Nightbug', 'リグル', 'リグル･ナイトバグ', 'リグルナイトバグ'),
    THC('ミスティア・ローレライ', 'Mystia Lorelei', 'ミスティア', 'ミスティア･ローレライ', 'ミスティアローレライ'),
    THC('上白沢慧音', 'Keine Kamishirasawa','慧音', '恵音'),
    THC('因幡てゐ', 'Tewi','てゐ'),
    THC('鈴仙・優曇華院・イナバ', 'Reisen Udongein Inaba','鈴仙・優曇華院', '鈴仙', '優曇華院', 'うどんげ', '鈴仙・U・イナバ', '鈴仙・優曇華・イナバ'),
    THC('八意永琳', 'Eirin Yagokoro', '永琳'),
    THC('蓬莱山輝夜', 'Kaguya Houraisan','輝夜'),
    THC('藤原妹紅', 'Fujiwara no Mokou', '妹紅', 'もこたん'),

    # PHANTASMAGORIA OF FLOWER VIEW
    THC('射命丸文', 'Aya Shameimaru', '文', '射命丸'),
    THC('メディスン・メランコリー', 'Medicine Melancholy', 'メディスン', 'メディスン･メランコリー', 'メディスンメランコリー', 'メディスン・メランコリ'),
    THC('風見幽香', 'Yuuka Kazami', '幽香'),
    THC('小野塚小町', 'Komachi Onozuka', '小町'),
    THC('四季映姫・ヤマザナドゥ', 'Eiki Shiki Yamaxanadu', 'ヤマザナドゥ', '四季映姫', '映姫', '四季映姫･ヤマザナドゥ', '四季映姫ヤマザナドゥ'),
    THC('リリーブラック', 'Lily Black'),

    # MOUNTAIN OF FAITH
    THC('秋静葉', 'Shizuha Aki', '静葉'),
    THC('秋穣子', 'Minoriko Aki', '穣子'),
    THC('鍵山雛', 'Hina Kagiyama', '雛'),
    THC('河城にとり', 'Nitori Kawashiro',  'にとり'),
    THC('犬走椛', 'Momiji Inubashiri', '椛'),
    THC('東風谷早苗', 'Sanae Kochiya', '早苗'),
    THC('八坂神奈子', 'Kanako Yasaka', '神奈子', '神奈子様'),
    THC('洩矢諏訪子', 'Suwako Moriya', '諏訪子', '諏訪子様', '守矢諏訪子'),

    # SCARLET WEATHER RAOHSODY
    THC('永江衣玖', 'Iku Nagae', '衣玖'),
    THC('比那名居天子', 'Tenshi Hinanawi', '天子'),

    # SUBTERRANEAN ANIMISM
    THC('キスメ', 'Kisume'),
    THC('黒谷ヤマメ', 'Yamame Kurodani', 'ヤマメ'),
    THC('水橋パルスィ', 'Parsee Mizuhashi', 'パルスィ'),
    THC('星熊勇儀', 'Yuugi Hoshiguma', '勇儀'),
    THC('古明地さとり', 'Satori Komeiji', 'さとり'),
    THC('火焔猫燐', 'Rin Kaenbyou', '燐', 'お燐'),
    THC('霊烏路空', 'Utsuho Reiuji', 'お空', '霊鳥路空'),
    THC('古明地こいし', 'Koishi Komeiji', 'こいし'),

    # UNIDENTIFIED FANTASTIC OBJECT
    THC('ナズーリン', 'Nazrin'),
    THC('多々良小傘', 'Kogasa Tatara', '小傘'),
    THC('雲居一輪', 'Ichirin Kumoi', '一輪'),
    THC('雲山', 'Unzan'),
    THC('村紗水蜜', 'Minamitsu Murasa', '水蜜'),
    THC('寅丸星', 'Shou Toramaru', '寅丸'),
    THC('聖白蓮', 'Hijiri Byakuren', '白蓮'),
    THC('封獣ぬえ', 'Nue Houjuu', 'ぬえ'),

    # DOUBLE SPOILER
    THC('姫海棠はたて', 'Hatate Himekaidou', 'はたて'),

    # FAIRY WARS
    THC('スターサファイア', 'Star Sapphire'),
    THC('ルナチャイルド', 'Luna Child'),
    THC('サニーミルク', 'Sunny Milk'),

    # TEN DESIRES
    THC('幽谷響子', 'Kyouko Kasodani', '響子'),
    THC('宮古芳香', 'Yoshika Miyako', '芳香'),
    THC('霍青娥', 'Seiga Kaku', '青娥'),
    THC('蘇我屠自古', 'Soga no Tojiko', '屠自古'),
    THC('物部布都', 'Mononobe no Futo', '布都'),
    THC('豊聡耳神子', 'Toyosatomimi no Miko', '神子'),
    THC('二ッ岩マミゾウ', 'Mamizou Futatsuiwa', '二ツ岩マミゾウ', 'マミゾウ'),

    # HOPELESS MASQUERADE
    THC('秦こころ', 'Hata no Kokoro', 'こころ'),
    THC('わかさぎ姫', 'Wakasagihime', 'わかさぎ'),
    THC('赤蛮奇', 'Sekibanki'),
    THC('今泉影狼', 'Kagerou Imaizumi', '影狼'),
    THC('九十九弁々', 'Benben Tsukumo', '弁々'),
    THC('九十九八橋', 'Yatsuhashi Tsukumo', '八橋'),
    THC('鬼人正邪', 'Seija Kijin', '正邪'),
    THC('少名針妙丸', 'Shinmyoumaru Sukuna', '針妙丸'),
    THC('堀川雷鼓', 'Raiko Horikawa', '雷鼓'),

    # URBAN LEGEND IN LIMBO
    THC('茨木華扇', 'Kasen Ibaraki', '華扇'),
    THC('宇佐見菫子', 'Sumireko Usami', '菫子'),

    # LEGACY OF LUNATIC KINGDOM
    THC('清蘭', 'Seiran'),
    THC('鈴瑚', 'Ringo'),
    THC('ドレミー・スイート', 'Doremy Sweet', 'ドレミー', 'ドレミー･スイート', 'ドレミースイート'),
    THC('稀神サグメ', 'Sagume Kishin', 'サグメ'),
    THC('クラウンピース', 'Clownpiece', 'Clownpiece'),
    THC('純狐', 'Junko', '純孤'),
    THC('ヘカーティア・ラピスラズリ', 'Hecatia Lapislazuli', 'へカーティア・ラピスラズリ',
        'ヘカーティア', 'へカーティア',
        'ヘカーティア･ラピスラズリ', 'へカーティア･ラピスラズリ',
        'ヘカーティアラピスラズリ', 'へカーティアラピスラズリ'),

    # ANTINOMY OF COMMON FLOWERS
    THC('依神女苑', 'Joon Yorigami', '女苑'),
    THC('依神紫苑', 'Shin Yorigami', '紫苑'),

    # HIDDEN STAR IN FOUR SEASONS
    THC('エタニティラルバ', 'Eternity Larvae'),
    THC('坂田ネムノ', 'Nemuno Sakata', 'ネムノ'),
    THC('高麗野あうん', 'Aunn Komano', 'あうん'),
    THC('矢田寺成美', 'Narumi Yatadera', '成美'),
    THC('丁礼田舞', 'Mai Teireida', '舞'),
    THC('爾子田里乃', 'Satono Nishida', '里乃'),
    THC('摩多羅隠岐奈', 'Okina Matara', '隠岐奈'),

    # WILY BEAST AND WEAKEST CREATURE
    THC('戎瓔花', 'Eika Ebisu'),
    THC('牛崎潤美', 'Urumi Ushizaki', '潤美'),
    THC('庭渡久侘歌', 'Kutaka Niwatari', '久侘歌', '庭渡久詫歌'),
    THC('吉弔八千慧', 'Yachie Kicchou', '八千慧', '八千恵', '吉兆八千慧'),
    THC('杖刀偶磨弓', 'Mayumi Joutouguu', '磨弓'),
    THC('埴安神袿姫', 'Keiki Haniyasushin', '袿姫'),
    THC('驪駒早鬼', 'Saki Kurokoma', '早鬼'),

    # UNCONNECTED MARKETEERS
    THC('豪徳寺ミケ', 'Mike Goutokuji', 'ミケ'),
    THC('山城たかね', 'Takane Yamashiro', 'たかね'),
    THC('駒草山如', 'Sannyo Komakusa', '山如'),

]

def loading_bar(done, left, done_i='=', left_i='-', fold=256, insert=']\n['):
    s = done_i*done + left_i*left
    return (''.join(l + insert * (n % fold == fold-1) for n, l in enumerate(s)))


def train_result(epoch, epoch_len, batch_num, batch_len,
                 loss, accuracy,
                 lr, timer, step):
    epoch, epoch_len = int(epoch) + 1, int(epoch_len)
    batch_num, batch_len, batch_left = int(batch_num) + 1, int(batch_len), int(batch_len-batch_num)
    loss, accuracy = float(loss), float(accuracy)
    print(
        f'EPOCH: {epoch}/{epoch_len}\n'
        f'Step Number: {step}\n'
        f'Batch Number: {batch_num}/{batch_len}\n'
        f'[{loading_bar(batch_num, batch_left)}]\n'
        f'Estimated Time Left: {timer}\n'
        f'Learning Rate: {lr:.3}\n'
        f'Loss:          {loss:.3}\n'
        f'Accuracy:      {accuracy:.3}\n'
    )

def test_result(epoch, epoch_len,
                 loss, accuracy, timer):
    epoch, epoch_len = int(epoch) + 1, int(epoch_len)
    loss, accuracy = float(loss), float(accuracy)
    print(
        'TEST RESULT\n' + '='*128 + '\n'
                                    f'EPOCH: {epoch}/{epoch_len}\n'
                                    f'Estimated Time Left: {timer}\n'
                                    f'Loss:     {loss:.3}\n'
                                    f'Accuracy: {accuracy:.3}\n'
    )

def spl_imwrite(directory, image):
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    im_buf_arr.tofile(directory)

def spl_imread(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    print(len(all_characters))
