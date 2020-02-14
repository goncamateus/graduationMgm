#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "fm_feature_extractor.h"
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/server_param.h>
#include <rcsc/player/world_model.h>
#include <rcsc/geom/sector_2d.h>
#include "strategy.h"

using namespace rcsc;

FMFeatureExtractor::FMFeatureExtractor(int num_teammates,
                                       int num_opponents,
                                       bool playing_offense) : FeatureExtractor(num_teammates, num_opponents, playing_offense)
{
  assert(numTeammates >= 0);
  assert(numOpponents >= 0);
  numFeatures = num_basic_features + features_per_teammate * numTeammates + features_per_opponent * numOpponents;
  // numFeatures+=2; // action status, stamina
  feature_vec.resize(numFeatures);
}

FMFeatureExtractor::~FMFeatureExtractor() {}

const std::vector<float> &
FMFeatureExtractor::ExtractFeatures(const rcsc::WorldModel &wm,
                                    bool last_action_status)
{
  featIndx = 0;
  const int COUNT_THR = 10;
  const ServerParam &SP = ServerParam::i();
  const SelfObject &self = wm.self();
  const Vector2D &self_pos = self.pos();
  const float self_ang = self.body().radian();
  const PlayerPtrCont &teammates = wm.teammatesFromSelf();
  const PlayerPtrCont &opponents = wm.opponentsFromSelf();
  float maxR = sqrtf(SP.pitchHalfLength() * SP.pitchHalfLength() + SP.pitchWidth() * SP.pitchWidth());
  // features about self pos
  // Allow the agent to go 10% over the playfield in any direction
  float tolerance_x = .1 * SP.pitchHalfLength();
  float tolerance_y = .1 * SP.pitchHalfWidth();
  Vector2D homePos = Strategy::i().getPosition(wm.self().unum());

  // Feature[0]: Able to kick
  addFeature(self.isKickable());

  // Feature[1]: X-postion
  addFeature(self_pos.x);

  // Feature[2]: Y-Position
  addFeature(self_pos.y);

  // Feature[3]: homePos dist
  addFeature(self_pos.dist(homePos));

  float r;
  float th;
  Vector2D ball_pos = wm.ball().pos();
  angleDistToPoint(self_pos, ball_pos, th, r);
  // Feature[4]: ball dist
  addFeature(r);
  // Feature[5]: relative angle to ball
  addFeature(th);

  double mates_congestion = get_player_congestion(wm.teammates(), self_pos);
  double opps_congestion = get_player_congestion(wm.opponents(), self_pos);
  // Feature[6]: teammates congestion
  addFeature(mates_congestion);
  // Feature[7]: opponents congestion
  addFeature(opps_congestion);

  const rcsc::AngleDeg ball_move_angle = (self_pos - ball_pos).th();
  const rcsc::Sector2D pass_cone(ball_pos,
                                 1.0, ball_pos.dist(self_pos) + 3.0, // radius
                                 ball_move_angle - 10.0, ball_move_angle + 10.0);

  // Feature[8]: opponents on pass course
  addFeature(wm.countOpponentsIn(pass_cone, COUNT_THR, true));

  const Sector2D front_space_sector(Vector2D(self_pos.x - 5.0, self_pos.y),
                                    4.0, 20.0, -15.0, 15.0);
  // Feature[9]: opponents in front
  addFeature(wm.countOpponentsIn(front_space_sector, COUNT_THR, false));

  // Features[10 - 9+T]: teammate's dist from self
  int detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it = wm.teammatesFromSelf().begin(); it != wm.teammatesFromSelf().end(); ++it)
  {
    const PlayerObject *teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates)
    {
      addFeature(self_pos.dist(teammate->pos()));

      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i = detected_teammates; i < numTeammates; ++i)
  {
    addFeature(FEAT_INVALID);
  }

  // Features[9+T - 8+T+O]: teammate's dist from self
  int detected_opps = 0;
  for (PlayerPtrCont::const_iterator it = wm.opponentsFromSelf().begin(); it != wm.opponentsFromSelf().end(); ++it)
  {
    const PlayerObject *opponent = *it;
    if (valid(opponent) && opponent->unum() > 0 && detected_opps < numOpponents)
    {
      addFeature(self_pos.dist(opponent->pos()));

      detected_opps++;
    }
  }
  // Add zero features for any missing teammates
  for (int i = detected_opps; i < numOpponents; ++i)
  {
    addFeature(FEAT_INVALID);
  }

  // Feature 9+T+O home pos x
  addFeature(homePos.x/pitchHalfLength);
  // Feature 10+T+O home pos y
  addFeature(homePos.y/pitchHalfWidth);

  assert(feature_vec.size() == numFeatures);
  // checkFeatures();
  return feature_vec;
}

bool FMFeatureExtractor::valid(const rcsc::PlayerObject *player)
{
  if (!player)
  {
    return false;
  } //avoid segfaults
  const rcsc::Vector2D &pos = player->pos();
  if (!player->posValid())
  {
    return false;
  }
  return pos.isValid();
}
