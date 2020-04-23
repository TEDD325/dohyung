/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.ScheduleData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

@Category(RequiresNone.class)
public class ScheduleDaoTest {

  private static final String TEST_ID = "123";

  @InjectMocks
  private ScheduleDao dao;

  @Mock
  private ScheduleRepository repos;

  @Mock
  private ScheduleEventRepository scheduleEventRepos;

  @Mock
  private DeviceServiceRepository deviceServiceRepository;

  private Schedule schedule;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    schedule = ScheduleData.newTestInstance();
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("Returned schedule is not null", dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrName() {
    when(repos.findOne(TEST_ID)).thenReturn(schedule);
    schedule.setId(TEST_ID);
    assertEquals("Returned schedule is not expected", schedule, dao.getByIdOrName(schedule));
  }

  @Test
  public void testGetByIdOrNameWithNoId() {
    when(repos.findByName(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(schedule);
    assertEquals("Returned schedule is not expected", schedule, dao.getByIdOrName(schedule));
  }

  @Test
  public void testIsScheduleAssociatedToScheduleEvent() {
    dao.isScheduleAssociatedToScheduleEvent(schedule);
  }

  @Test
  public void testIsScheduleAssociatedToScheduleEventWithNull() {
    assertFalse("Null should not return association to even",
        dao.isScheduleAssociatedToScheduleEvent(null));
  }

  @Test
  public void testGetAffectedServices() {
    List<ScheduleEvent> events = new ArrayList<>();
    events.add(ScheduleEventData.newTestInstance());
    when(scheduleEventRepos.findBySchedule(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(events);
    dao.getAffectedServices(schedule);
  }

}
