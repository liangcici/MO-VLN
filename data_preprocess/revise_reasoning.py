import os
import json
import argparse


def generate_data(args):
    # gt_path = os.path.join('dataset/reasoning', args.scene + '_instruction_0605_inference_GPT4_checked_2.json')
    # if args.model_type == 'GPT4':
    #     data_path = os.path.join('dataset/reasoning', args.scene + '_instruction_0605_inference_whmatches.json')
    # else:
    data_path = os.path.join('dataset/reasoning', args.scene + '_instruction_0606_inference_{}.json'.format(args.model_type))
    objectnav_path = os.path.join('dataset/objectnav', args.scene + '.json')
    categories_path = os.path.join('dataset/objectnav', args.scene + '_categories.json')

    # with open(gt_path, 'r') as f:
    #     gt_data = json.load(f)
    with open(data_path, 'r') as f:
        gen_data = json.load(f)
    with open(objectnav_path, 'r') as f:
        object_data = json.load(f)
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    goal_to_id = {}
    for i, cat in enumerate(categories):
        goal_to_id[cat] = i

    if args.model_type == 'GPT4':
        item_name = 'matching_options'
    else:
        item_name = 'matching_options_{}'.format(args.model_type)

    new_data = []
    gen_index = 0
    # for i, item in enumerate(gt_data):
    for i, item in enumerate(gen_data):
        instr = item['instruction']
        try:
            assert instr == gen_data[gen_index]['instruction']
        except Exception as e:
            print('gt instr: ', instr)
            print('llm instr: ', gen_data[gen_index]['instruction'])
            # assert gen_data[gen_index]['instruction'].strip() == ''
            gen_index += 1

        gt_matched_objects = []
        for cat in item['objects_matches_GPT4_checked']:
            if cat in categories:
                gt_matched_objects.append(cat)

        if len(gt_matched_objects) == 0:
            continue

        gt_matched_obj_ids = []
        for obj in gt_matched_objects:
            gt_matched_obj_ids.append(goal_to_id[obj])

        pred_goals = []
        pred_goal_ids = []
        for obj in gen_data[gen_index][item_name]:
            if obj in goal_to_id:
                pred_goals.append(obj)
                pred_goal_ids.append(goal_to_id[obj])
        if len(pred_goals) > 0:
            pred_goals = pred_goals[0]
            pred_goal_ids = pred_goal_ids[0]
        else:
            pred_goals = None
            pred_goal_ids = None

        # nearest path
        shortest_dist = 10000
        best_goal = None
        for cat in gt_matched_objects:
            for data_item in object_data:
                if data_item['goal']['name'].lower() == cat:
                    if data_item['goal']['shortest_distance'] < shortest_dist:
                        best_goal = data_item['goal']
                    break

        assert best_goal is not None
        new_item = {}
        new_item['episode_id'] = i
        new_item['scene'] = object_data[0]['scene']
        new_item['start_position'] = object_data[0]['start_position']
        new_item['instruction'] = instr
        new_item['goals'] = gt_matched_objects
        new_item['goal_ids'] = gt_matched_obj_ids
        new_item['pred_goals'] = pred_goals
        new_item['pred_goal_ids'] = pred_goal_ids
        new_item['best_goal'] = best_goal

        print(new_item)
        new_data.append(new_item)
        gen_index += 1

    with open(os.path.join(args.save_path, args.scene + '_{}.json'.format(args.model_type)), 'w') as f:
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
        default="dataset/reasoning",
        help="path to load dataset",
    )
    parser.add_argument(
        "--save_path",
        default="dataset/reasoning",
        help="path to save dataset",
    )
    parser.add_argument(
        "--model_type",
        default="GPT4",
        help="data type",
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
