-- done method
function game_done()
	local done = (is_dead() or is_boss_dead())

	if is_boss_dead() then
		data.beat_stage = true
	end

	return done
end

-- reward function
function game_reward()
	local reward = 0

	-- reward for finishing
	if is_boss_dead() then
		return clear_reward
	end

	-- punishment for dying
	if is_dead() then
		return -death_punishment
	end


	-- health rewards
	delta_health = data.health - prev_health
	if delta_health > 0 then
		reward = reward + (delta_health * health_gain_reward)
	else
		reward = reward + (delta_health * health_loss_punishment)
	end


-- -- -- -- position rewards -- -- -- --

	delta_posX = data.posX - prev_posX
	delta_posY = -(data.posY - prev_posY) 	-- negate to make positive => going up

    -- reward progress if not at boss
    if not is_at_boss() and data.posX > max_posX then
        reward = reward + (delta_posX * progress_reward)
    end

-- default

	-- -- -- -- -- -- -- -- -- -- -- -- -- --

	-- reward killing stuff
	reward = reward + enemy_death_bonus()

    -- reward hitting the boss
    reward = reward + boss_damage_bonus()

	-- time punishment
	reward = reward - time_punishment

	-- punish shot count increasing
	reward = reward - shot_count_bonus()

	update_storage()

	-- clip reward
	if reward > clip_reward then
		reward = clip_reward
	elseif reward < -clip_reward then
		reward = -clip_reward
	end

	return reward

end


-- returns bonus reward upon killing an enemy
function enemy_death_bonus()

	local bonus = 0

	if data.enemy1_health < enemy1_prev_health and data.enemy1_health == 0 then
		bonus = bonus + kill_reward
	elseif data.enemy2_health < enemy2_prev_health and data.enemy2_health == 0 then
		bonus = bonus + kill_reward
	elseif data.enemy3_health < enemy3_prev_health and data.enemy3_health == 0 then
		bonus = bonus + kill_reward
	elseif data.enemy4_health < enemy4_prev_health and data.enemy4_health == 0 then
		bonus = bonus + kill_reward
	end

	return bonus

end


-- return a small punishment every time the number of bullets on screen incereases (to discourage shot spam)
function shot_count_bonus()
	if data.num_shots > prev_num_shots then
		return shot_punishment
	end
	return 0
end

-- returns bonus reward for damaging boss
function boss_damage_bonus()

    -- I believe boss is always at enemy id 1 (enemy healths seem to get set to 0 when entering the pre-boss room so theres space for the boss in RAM, so the boss spawns in id 1 as a result)
    if data.enemy1_id == boss_id and data.enemy1_health < enemy1_prev_health then
        local damage = (enemy1_prev_health - data.enemy1_health)

        if damage == 1 then
            return boss_low_damage_reward
        else
            return boss_high_damage_reward
        end
    end

    return 0
end

-- return true if boss is dead
function is_boss_dead()
    return (data.enemy1_id == boss_id and data.enemy1_health == 0 and enemy1_prev_health > 0)
end

-- return true if in the boss room
function is_at_boss()
    return data.posX > 7960
end

-- return true if X is dead lol
function is_dead()
	return (data.health == 0)
end


-- update stored variables
function update_storage()

	-- update local storage
	if data.health < prev_health then
		damage_taken = damage_taken + (prev_health - data.health)
	end

	prev_posX = data.posX
	prev_posY = data.posY
	prev_health = data.health

	enemy1_prev_health = data.enemy1_health
	enemy2_prev_health = data.enemy2_health
	enemy3_prev_health = data.enemy3_health
	enemy4_prev_health = data.enemy4_health

	prev_num_shots = data.num_shots

	if data.posX > max_posX then
		max_posX = data.posX
	end

	-- update log variables
	data.damage_taken = damage_taken
	data.furthest_position = max_posX

end

--------------- storage (common between files)

prev_posX = 128
prev_posY = 671

max_posX = 128
max_posY = 671

prev_health = 16

damage_taken = 0

enemy1_prev_health = 0
enemy2_prev_health = 0
enemy3_prev_health = 0
enemy4_prev_health = 0

prev_num_shots = 0

--------------- parameters

-- reward for furthest x position increasing
progress_reward = 0.2

-- reward / punishment for gaining / losing height where relevant
height_gain_reward = 0.05
height_loss_punishment = 0.07

-- general punishment
time_punishment = 0.05

-- reward / punishment for gaining / losing health
health_gain_reward = 1
health_loss_punishment = 5

-- reward for killing an enemy
kill_reward = 5

-- punishment for shooting (not for pressing the shoot button!) (keep VERY small or agent will decide to never shoot)
shot_punishment = 0.01

-- rewards for dealing 1 or >1 damage to the boss
boss_low_damage_reward = 15
boss_high_damage_reward = 15

-- reward for beating level (this doesnt get clipped)
clear_reward = 15

-- punishment for dying (also doesnt get clipped)
death_punishment = 15

-- clip
clip_reward = 15

--------------- unique storage

boss_id = 10 -- enemy id for sting chameleon