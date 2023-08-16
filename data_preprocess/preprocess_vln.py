import os
import json
import argparse
import sys
sys.path.append('./data_preprocess/')

from preprocess_simpleinstr import calc_similarity


def generate_data(args):

    # gen_data_path = os.path.join(args.data_path, args.scene + '_trajectory_VLN_landmark_{}.json'.format(args.model_type))
    gen_data_path = os.path.join(args.data_path, '{}_VLN_{}.json'.format(args.model_type, args.map_id))
    categories_path = os.path.join('dataset/objectnav', args.scene + '_categories.json')

    with open(gen_data_path, 'r') as f:
        gen_data = json.load(f)
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    goal_to_id = {}
    for i, cat in enumerate(categories):
        goal_to_id[cat] = i

    with open('data_preprocess/name_replace_dict.json', 'r') as f:
        replace_name = json.load(f)
    if args.scene == 'TG':
        with open('data_preprocess/TG_objectname_replaced.json', 'r') as f:
            replace_obj_name = json.load(f)
    else:
        replace_obj_name = None

    if replace_obj_name is not None:
        for k in replace_obj_name:
            if k in replace_name:
                replace_name[k] = replace_obj_name[k]

    if args.with_action:
        item_name = 'landmark_with_action_{}'.format(args.model_type)
        instr_name = 'instruction_with_action'
    else:
        item_name = 'landmark_without_action_{}'.format(args.model_type)
        instr_name = 'instruction_withoutaction'

    new_data = []
    eps_id = 0
    for ind, item in enumerate(gen_data):
        instr = item[instr_name]
        goal_name = item['object_name']
        if goal_name in replace_name:
            new_name = replace_name[goal_name]
        elif replace_obj_name is not None:
            new_name = replace_obj_name[goal_name]
        else:
            new_name = goal_name

        if new_name.lower() not in categories:
            print(new_name)
            continue

        new_item = {}
        new_item['episode_id'] = eps_id
        new_item['scene'] = args.scene
        new_item['instr'] = instr
        new_item['start_position'] = item['trajectory'][0]
        new_item['start_position'].append(0.)   # yaw
        new_item['goal'] = {}
        new_item['goal']['name'] = new_name
        new_item['goal']['cat_id'] = goal_to_id[new_name.lower()]
        new_item['goal']['best_position'] = item['target_object_position'][:3]
        new_item['goal']['shortest_distance'] = item['length_min']

        pred_goals = []
        pred_goal_ids = []
        for obj in item[item_name]:
            obj_name = obj.lower()
            if obj_name in goal_to_id:
                pred_goals.append(obj)
                pred_goal_ids.append(goal_to_id[obj_name])
            else:
                find_goal = False
                for cat in categories:
                    if cat in obj_name:
                        find_goal = True
                        pred_goals.append(cat)
                        pred_goal_ids.append(goal_to_id[cat])
                        break

                if find_goal:
                    continue
                max_score = 0.
                for cat in categories:
                    score = calc_similarity(cat, obj_name)
                    if score > max_score:
                        max_score = score
                        pred_cat = cat
                        if score > 0.8:
                            break
                if max_score > 0.5:
                    pred_goals.append(pred_cat)
                    pred_goal_ids.append(goal_to_id[pred_cat])

        new_item['pred_goal'] = pred_goals
        new_item['pred_goal_id'] = pred_goal_ids
        new_data.append(new_item)
        eps_id += 1

        print(new_item)

    if args.with_action:
        file_name = args.scene + '_with_action_{}.json'.format(args.model_type)
    else:
        file_name = args.scene + '_without_action_{}.json'.format(args.model_type)
    with open(os.path.join(args.save_path, file_name), 'w') as f:
        json.dump(new_data, f)


def main():
    parser = argparse.ArgumentParser(description="Generate ObjectNav Dataset")
    parser.add_argument(
        '--map_id', type=int, default=3,
        help="3: Starbucks; 4: TG; 5: NursingRoom"
    )
    parser.add_argument(
        "--scene",
        default="TG",
        help="scene type, [TG, XBK, YLY]",
    )
    parser.add_argument(
        "--data_path",
        default="dataset/vln",
        help="path to load dataset",
    )
    parser.add_argument(
        "--save_path",
        default="dataset/vln",
        help="path to save dataset",
    )
    parser.add_argument(
        "--model_type",
        default="chatGPT",
        help="data type",
    )
    parser.add_argument(
        "--with_action", type=int,
        default=0,
        help="0: w/o action, 1 w/ action",
    )
    args = parser.parse_args()

    if args.map_id == 3:
        args.scene = 'Starbucks'
    elif args.map_id == 4:
        args.scene = 'TG'
    elif args.map_id == 5:
        args.scene = 'NursingRoom'

    generate_data(args)


if __name__ == '__main__':
    main()


