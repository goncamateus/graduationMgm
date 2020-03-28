#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "highlevel_feature_extractor.h"
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/ray_2d.h>
#include "strategy.h"

using namespace rcsc;

HighLevelFeatureExtractor::HighLevelFeatureExtractor(int num_teammates,
                                                     int num_opponents,
                                                     bool playing_offense) : FeatureExtractor(num_teammates, num_opponents, playing_offense)
{
  assert(numTeammates >= 0);
  assert(numOpponents >= 0);
  numFeatures = num_basic_features + features_per_teammate * numTeammates + features_per_opponent * numOpponents;
  // numFeatures+=2; // action status, stamina
  feature_vec.resize(numFeatures);
}

HighLevelFeatureExtractor::~HighLevelFeatureExtractor() {}

const std::vector<float> &
HighLevelFeatureExtractor::ExtractFeatures(const rcsc::WorldModel &wm,
                                           bool last_action_status)
{
  featIndx = 0;
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
  // Feature[0]: X-postion
  addFeature(self_pos.x);

  // Feature[1]: Y-Position
  addFeature(self_pos.y);

  // Feature[2]: Self Angle
  addFeature(self_ang);

  float r;
  float th;
  // Features about the ball
  Vector2D ball_pos = wm.ball().pos();
  angleDistToPoint(self_pos, ball_pos, th, r);
  // Feature[3] and [4]: (x,y) postition of the ball
  if (playingOffense)
  {
    addFeature(ball_pos.x);
  }
  else
  {
    addFeature(ball_pos.x);
  }
  addFeature(ball_pos.y);
  // Feature[5]: Able to kick
  addFeature(self.isKickable());

  // Features about distance to goal center
  Vector2D goalCenter(SP.pitchHalfLength(), 0);
  if (!playingOffense)
  {
    goalCenter.assign(-SP.pitchHalfLength(), 0);
  }
  angleDistToPoint(self_pos, goalCenter, th, r);
  // Feature[6]: Goal Center Distance
  addFeature(r);
  // Feature[7]: Angle to goal center
  addFeature(th);
  // Feature[8]: largest open goal angle
  addFeature(calcLargestGoalAngle(wm, self_pos));
  // Feature[9]: Dist to our closest opp
  if (numOpponents > 0)
  {
    calcClosestOpp(wm, self_pos, th, r);
    addFeature(r);
  }
  else
  {
    addFeature(FEAT_INVALID);
  }

  // Features[10 - 10+T]: teammate's open angle to goal
  int detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it = teammates.begin(); it != teammates.end(); ++it)
  {
    const PlayerObject *teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates)
    {
      addFeature(calcLargestGoalAngle(wm, teammate->pos()));
      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i = detected_teammates; i < numTeammates; ++i)
  {
    addFeature(FEAT_INVALID);
  }

  // Features[10+T - 10+2T]: teammates' dists to closest opps
  if (numOpponents > 0)
  {
    detected_teammates = 0;
    for (PlayerPtrCont::const_iterator it = teammates.begin(); it != teammates.end(); ++it)
    {
      const PlayerObject *teammate = *it;
      if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates)
      {
        calcClosestOpp(wm, teammate->pos(), th, r);
        addFeature(r);
        detected_teammates++;
      }
    }
    // Add zero features for any missing teammates
    for (int i = detected_teammates; i < numTeammates; ++i)
    {
      addFeature(FEAT_INVALID);
    }
  }
  else
  { // If no opponents, add invalid features
    for (int i = 0; i < numTeammates; ++i)
    {
      addFeature(FEAT_INVALID);
    }
  }

  // Features [10+2T - 10+3T]: open angle to teammates
  detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it = teammates.begin(); it != teammates.end(); ++it)
  {
    const PlayerObject *teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates)
    {
      addFeature(calcLargestTeammateAngle(wm, self_pos, teammate->pos()));
      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i = detected_teammates; i < numTeammates; ++i)
  {
    addFeature(FEAT_INVALID);
  }

  // Features [10+3T - 10+6T]: x, y, unum of teammates
  detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it = teammates.begin(); it != teammates.end(); ++it)
  {
    const PlayerObject *teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates)
    {
      if (playingOffense)
      {
        addFeature(teammate->pos().x);
      }
      else
      {
        addFeature(teammate->pos().x);
      }
      addFeature(teammate->pos().y);
      addFeature(teammate->unum());
      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i = detected_teammates; i < numTeammates; ++i)
  {
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
  }

  // Features [10+6T - 10+6T+3O]: x, y, unum of opponents
  int detected_opponents = 0;
  for (PlayerPtrCont::const_iterator it = opponents.begin(); it != opponents.end(); ++it)
  {
    const PlayerObject *opponent = *it;
    if (valid(opponent) && opponent->unum() > 0 && detected_opponents < numOpponents)
    {
      if (playingOffense)
      {
        addFeature(opponent->pos().x);
      }
      else
      {
        addFeature(opponent->pos().x);
      }
      addFeature(opponent->pos().y);
      addFeature(opponent->unum());
      detected_opponents++;
    }
  }
  // Add zero features for any missing opponents
  for (int i = detected_opponents; i < numOpponents; ++i)
  {
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
  }

  const int self_min = wm.interceptTable()->selfReachCycle();
  const int mate_min = wm.interceptTable()->teammateReachCycle();
  const int opp_min = wm.interceptTable()->opponentReachCycle();

  bool isTackle = false;
  double tackle_prob = wm.self().tackleProbability();

  if (wm.self().card() == NO_CARD && (wm.ball().pos().x > SP.ourPenaltyAreaLineX() + 0.5 || wm.ball().pos().absY() > SP.penaltyAreaHalfWidth() + 0.5) && tackle_prob < wm.self().foulProbability())
  {
    tackle_prob = wm.self().foulProbability();
  }

  if (tackle_prob > 0.8)
  {

    const Vector2D self_reach_point = wm.ball().inertiaPoint(self_min);

    //
    // check where the ball shall be gone without tackle
    //

    bool ball_will_be_in_our_goal = false;

    if (self_reach_point.x < -SP.pitchHalfLength())
    {
      const Ray2D ball_ray(wm.ball().pos(), wm.ball().vel().th());
      const Line2D goal_line(Vector2D(-SP.pitchHalfLength(), 10.0),
                             Vector2D(-SP.pitchHalfLength(), -10.0));

      const Vector2D intersect = ball_ray.intersection(goal_line);
      if (intersect.isValid() && intersect.absY() < SP.goalHalfWidth() + 1.0)
      {
        ball_will_be_in_our_goal = true;
      }
    }

    if (wm.existKickableOpponent() || ball_will_be_in_our_goal || (opp_min < self_min - 3 && opp_min < mate_min - 3) || (self_min >= 5 && wm.ball().pos().dist2(SP.theirTeamGoalPos()) < std::pow(10.0, 2) && ((SP.theirTeamGoalPos() - wm.self().pos()).th() - wm.self().body()).abs() < 45.0))
    {
      isTackle = true;
    }
  }

  addFeature(isTackle);
  addFeature(self.stamina());

  assert(featIndx == numFeatures);
  // checkFeatures();
  return feature_vec;
}

bool HighLevelFeatureExtractor::valid(const rcsc::PlayerObject *player)
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
