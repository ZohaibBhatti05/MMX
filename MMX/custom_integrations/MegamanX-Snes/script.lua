-- done method
function contest_done()
	local done = (is_dead() or is_grabbed_by_vile())

	if is_grabbed_by_vile() then
		data.beat_stage = true
	end

	return done
end
	
-- reward function
function contest_reward()
	local reward = 0

	-- reward for finishing
	if is_grabbed_by_vile() then
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


	if is_in_bee_area() then
-- if at the bee section

		-- reward height gain / punish falling
		if delta_posY > 0 then
			reward = reward + (delta_posY * height_gain_reward)
		-- punish falling
		elseif delta_posY < 0 then
			reward = reward + (delta_posY * height_loss_punishment)
		end

	elseif not is_at_vile() then
-- if NOT at the bee section and not near the end

		-- reward progress
		if data.posX > max_posX then
			reward = reward + (delta_posX * progress_reward)
		end
	
	end


-- default

	-- -- -- -- -- -- -- -- -- -- -- -- -- --
	
	-- reward killing stuff	
	reward = reward + enemy_death_bonus()

	-- time punishment
	reward = reward - time_punishment

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

-- return true if X is dead lol
function is_dead()
	return (data.health == 0)
end
	
-- return true if X is being grabbed by Vile (at the end of the intro stage)
function is_grabbed_by_vile()
	return (data.X_state == 48)
end

-- return true if X is in area with the bee miniboss
function is_in_bee_area()
	x = data.posX
	return ((2700 < x and x < 2800) or (3320 < x and x < 3430))
end

-- return true if at the bit where you kill 3 dudes then fight Vile
function is_at_vile()
	return (data.posX > 7100)
end

-- return true if Vile is on screen
function is_vile_onscreen()
	if data.enemy1_id == 50 then
		return true
	elseif data.enemy2_id == 50 then
		return true
	elseif data.enemy3_id == 50 then
		return true
	elseif data.enemy4_id == 50 then
		return true
	end
	
	return false
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
	
	if data.posX > max_posX then
		max_posX = data.posX
	end

	-- update log variables
	data.damage_taken = damage_taken
	data.furthest_position = max_posX
	
	-- change health reward if vile is on screen
	if is_vile_onscreen() and near_end == false then
		health_loss_punishment = -health_loss_punishment -- reward damage if vile on screen
		near_end = true
	end
end

--------------- storage

prev_posX = 128
prev_posY = 367

max_posX = 128
max_posY = 367

prev_health = 16

damage_taken = 0

enemy1_prev_health = 0
enemy2_prev_health = 0
enemy3_prev_health = 0
enemy4_prev_health = 0

--------------- parameters

near_end = false

-- reward for furthest x position increasing
progress_reward = 0.2

-- reward / punishment for gaining / losing height
height_gain_reward = 0.05
height_loss_punishment = 0.07

-- general punishment
time_punishment = 0.05

-- reward / punishment for gaining / losing health
health_gain_reward = 1
health_loss_punishment = 5

-- reward for killing an enemy
kill_reward = 5

-- reward for beating level (this doesnt get clipped)
clear_reward = 15

-- punishment for dying (also doesnt get clipped)
death_punishment = 15

-- clip
clip_reward = 15
